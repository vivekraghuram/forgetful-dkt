import numpy as np
import tensorflow as tf
from sklearn import metrics

class LSTM(object):

  def __init__(self,
               hidden_dim=512,
               output_dim=1004,
               input_dim=4096,
               learning_rate=5e-5,
               batch_size=50,
               num_layers=1):

    self.hidden_dim = hidden_dim
    self.output_dim = output_dim
    self.input_dim = input_dim
    self.learning_rate = learning_rate
    self.batch_size = batch_size
    self.num_layers = num_layers

    self.sy_keep_prob = tf.placeholder(tf.float32, shape=[], name="keep_prob")
    self.sy_input = tf.placeholder(shape=[self.batch_size, None, self.input_dim], name="input", dtype=tf.float32)
    self.sy_target = tf.placeholder(shape=[self.batch_size, None, self.output_dim], name="target", dtype=tf.float32)
    self.sy_target_mask = tf.placeholder(shape=[self.batch_size, None, self.output_dim], name="mask_not_null", dtype=tf.float32)

    self.sy_hidden_states = [None] * self.num_layers
    self.sy_cell_states = [None] * self.num_layers
    for i in range(self.num_layers):
      self.sy_hidden_states[i] = tf.placeholder(shape=[self.batch_size, self.hidden_dim], name="hidden_state-%i"%i, dtype=tf.float32)
      self.sy_cell_states[i] = tf.placeholder(shape=[self.batch_size, self.hidden_dim], name="cell_state-%i"%i, dtype=tf.float32)

  def build_model(self, activation=None):
    initial_state = tuple([tf.nn.rnn_cell.LSTMStateTuple(
                                            self.sy_hidden_states[i],
                                            self.sy_cell_states[i]
                                          ) for i in range(self.num_layers)])

    cell = tf.contrib.rnn.LSTMCell(self.hidden_dim)
    cell = tf.contrib.rnn.MultiRNNCell([cell for _ in range(self.num_layers)], state_is_tuple=True)

    self.output_hidden_state, self.output_cell_state = tf.nn.dynamic_rnn(cell, self.sy_input, initial_state=initial_state)

    logits = tf.layers.dense(inputs=self.output_hidden_state, units=self.output_dim, activation=activation)

    # Apply target mask as weights to zero out meaningless values
    self.mse = tf.losses.mean_squared_error(self.sy_target, logits, weights=self.sy_target_mask)
    self.update_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.mse)
    self.predictions = tf.round(logits)


  def train(self, sess, data, epochs=10, encode=lambda x,y: y):
    # Assumes data is an object where training_batches is an iterator over all the data and it yields
    # a tuple consisting of inputs, target and target masks
    for e in range(epochs):
      for batch in data.training_batches(self.batch_size):
        inputs, targets, target_masks = batch
        feed_dict = {
          self.sy_input       : encode(sess, inputs),
          self.sy_target      : targets,
          self.sy_target_mask : target_masks
        }

        for i in range(self.num_layers):
          feed_dict[self.sy_hidden_states[i]] = np.zeros((self.batch_size, self.hidden_dim))
          feed_dict[self.sy_cell_states[i]] = np.zeros((self.batch_size, self.hidden_dim))

        _, c = sess.run([self.update_op, self.mse], feed_dict=feed_dict)

      print("epoch %d, MSE: %.4f" % (e, c))

  def test(self, sess, data, encode=lambda x,y: y):
    accuracy = 0.0
    baseline = 0.0
    num_predictions = 0
    all_predictions = []
    all_targets = []

    for batch in data.testing_batches(self.batch_size):
      inputs, targets, target_masks = batch
      feed_dict = {
        self.sy_input       : encode(sess, inputs),
        self.sy_target      : targets,
        self.sy_target_mask : target_masks
      }

      for i in range(self.num_layers):
        feed_dict[self.sy_hidden_states[i]] = np.zeros((self.batch_size, self.hidden_dim))
        feed_dict[self.sy_cell_states[i]] = np.zeros((self.batch_size, self.hidden_dim))

      p, hs, cs = sess.run([self.predictions, self.output_hidden_state, self.output_cell_state],
                            feed_dict=feed_dict)

      accuracy += np.sum(np.equal(targets, p) * (target_masks != 0), axis=None)
      baseline += np.sum(targets)
      num_predictions += np.sum(target_masks != 0)
      all_predictions.extend(np.sum(p * (target_masks != 0), axis=2).reshape((p.shape[0] * p.shape[1])))
      all_targets.extend(np.sum(targets, axis=2).reshape(targets.shape[0] * targets.shape[1]))

    auc = metrics.roc_auc_score(all_targets, all_predictions)
    baseline_score = max(baseline/num_predictions, 1.0 - baseline/num_predictions)
    print("Accuracy: %.4f, Baseline: %.4f, AUC: %.5f" % (accuracy/num_predictions, baseline_score, auc))
    return accuracy/num_predictions, baseline_score
