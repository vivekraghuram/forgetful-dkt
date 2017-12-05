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
    self.sy_dropout = tf.placeholder(tf.float32)

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

    cell = tf.contrib.rnn.MultiRNNCell([tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.LSTMCell(self.hidden_dim),
                                                                      output_keep_prob=1.0 - self.sy_dropout) for _ in range(self.num_layers)],
                                       state_is_tuple=True)

    self.output_hidden_state, self.output_cell_state = tf.nn.dynamic_rnn(cell, self.sy_input, initial_state=initial_state)

    self.all_logits = tf.layers.dense(inputs=self.output_hidden_state, units=self.output_dim)
    self.logits = tf.reduce_sum(self.all_logits * self.sy_target_mask, axis=2)
    self.prob = tf.nn.sigmoid(self.logits)

    # Apply target mask as weights to zero out meaningless values
    self.loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.reduce_sum(self.sy_target, axis=2), logits=self.logits)
    self.update_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
    self.predictions = tf.round(self.prob)


  def train(self, sess, data, epochs=10, dropout_prob=0.5, encode=lambda x,y: y, verbose=False):
    # Assumes data is an object where training_batches is an iterator over all the data and it yields
    # a tuple consisting of inputs, target and target masks
    for e in range(epochs):
      for batch in data.training_batches(self.batch_size):
        inputs, targets, target_masks = batch
        feed_dict = {
          self.sy_input       : encode(sess, inputs),
          self.sy_target      : targets,
          self.sy_target_mask : target_masks,
          self.sy_dropout     : dropout_prob
        }

        for i in range(self.num_layers):
          feed_dict[self.sy_hidden_states[i]] = np.zeros((self.batch_size, self.hidden_dim))
          feed_dict[self.sy_cell_states[i]] = np.zeros((self.batch_size, self.hidden_dim))

        _, c = sess.run([self.update_op, self.loss], feed_dict=feed_dict)

      if verbose:
        print("epoch %d, CE: %.4f" % (e, np.sum(c) / np.sum(target_masks != 0)))

  def test(self, sess, data, encode=lambda x,y: y, use_auc=True):
    accuracy = 0.0
    baseline = 0.0
    mae = 0.0
    cross_entropy = 0.0
    num_predictions = 0
    all_predictions = []
    all_targets = []

    for batch in data.testing_batches(self.batch_size):
      inputs, targets, target_masks = batch
      feed_dict = {
        self.sy_input       : encode(sess, inputs),
        self.sy_target      : targets,
        self.sy_target_mask : target_masks,
        self.sy_dropout     : 0.0
      }

      for i in range(self.num_layers):
        feed_dict[self.sy_hidden_states[i]] = np.zeros((self.batch_size, self.hidden_dim))
        feed_dict[self.sy_cell_states[i]] = np.zeros((self.batch_size, self.hidden_dim))

      a, p, ce = sess.run([self.predictions, self.prob, self.loss], feed_dict=feed_dict)

      accuracy += np.sum(np.equal(np.sum(targets, axis=2), a) * np.sum(target_masks != 0, axis=2), axis=None)
      baseline += np.sum(targets)
      mae += np.sum(np.absolute(np.sum(targets, axis=2) - p) * np.sum(target_masks != 0, axis=2), axis=None)
      num_predictions += np.sum(target_masks != 0)
      cross_entropy += np.sum(ce)

      summed_probs = p
      summed_targets = np.sum(targets, axis=2)
      summed_masks = np.sum(target_masks != 0, axis=2)
      for idx in range(len(summed_probs)):
        all_predictions.extend(summed_probs[idx, 0:int(np.sum(summed_masks[idx]))])
        all_targets.extend(summed_targets[idx, 0:int(np.sum(summed_masks[idx]))])

    assert(len(all_predictions) == num_predictions)
    assert(len(all_targets) == num_predictions)

    baseline_score = max(baseline/num_predictions, 1.0 - baseline/num_predictions)
    if use_auc:
      auc = metrics.roc_auc_score(all_targets, all_predictions)
      print("Accuracy: %.6f, Baseline: %.6f, AUC: %.6f, MAE: %.6f, CE: %.6f" % (accuracy/num_predictions, baseline_score, auc, mae/num_predictions, cross_entropy/num_predictions))
      return accuracy/num_predictions, baseline_score, auc, mae/num_predictions, ce/num_predictions

    print("Accuracy: %.6f, Baseline: %.6f, MAE: %.6f, CE: %.6f" % (accuracy/num_predictions, baseline_score, mae/num_predictions, cross_entropy/num_predictions))
    return accuracy/num_predictions, baseline_score, mae/num_predictions, ce/num_predictions
