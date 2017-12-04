import numpy as np
import tensorflow as tf

class Encoder(object):

  def __init__(self,
               input_dim,
               encoded_dim,
               layer_config=None,
               activation=tf.tanh,
               output_activation=None,
               learning_rate=5e-3,
               batch_size=128,
               scope="encoder"):

    if layer_config is None:
      layer_config = ()

    assert(type(layer_config) == tuple)

    self.batch_size = batch_size
    self.input_dim = input_dim
    self.encoded_dim = encoded_dim

    with tf.variable_scope(scope):
      self.sy_input = tf.placeholder(tf.float32, [None, input_dim], "input")
      self.sy_target = tf.placeholder(tf.float32, [None, input_dim], "target")
      prev_layer = self.sy_input

      for dim in layer_config:
        prev_layer = tf.layers.dense(inputs=prev_layer, units=dim, activation=activation)

      self.encoded_out = tf.layers.dense(inputs=prev_layer, units=encoded_dim, activation=output_activation)
      prev_layer = self.encoded_out

      for dim in layer_config[::-1]:
        prev_layer = tf.layers.dense(inputs=prev_layer, units=dim, activation=activation)

      self.decoded_out = tf.layers.dense(inputs=prev_layer, units=input_dim, activation=output_activation)

      self.mse = tf.losses.mean_squared_error(self.sy_target, self.decoded_out)
      self.update_op = tf.train.AdamOptimizer(learning_rate).minimize(self.mse)

  def train(self, sess, data, epochs=1, num_sequences_per_iter=None):
    if num_sequences_per_iter is None:
      num_sequences_per_iter = self.batch_size
    for e in range(epochs):
      data.shuffle()
      cost = []
      for i, (inputs, targets, target_masks) in enumerate(data.training_batches(num_sequences_per_iter)):
        vectorized_inputs = inputs.reshape((inputs.shape[0] * inputs.shape[1], inputs.shape[2]))
        vectorized_inputs = vectorized_inputs[np.sum(vectorized_inputs, axis=1) != 0]

        start_idx, end_idx = 0, self.batch_size
        iter_costs = []
        while end_idx < vectorized_inputs.shape[0]:

          feed_dict = {
            self.sy_input       : vectorized_inputs[start_idx:end_idx],
            self.sy_target      : vectorized_inputs[start_idx:end_idx]
          }

          _, c = sess.run([self.update_op, self.mse], feed_dict=feed_dict)
          start_idx, end_idx = end_idx, end_idx + self.batch_size
          cost.append(c)
          iter_costs.append(c)

        print("Iteration %d, MSE: %.4f" % (i, np.mean(iter_costs)))

      print("epoch %d, MSE: %.4f" % (e, np.mean(cost)))

  def encode(self, sess, array):
    """ Array is N x input_dim and the output will be N x encoded_dim and N x input_dim"""
    assert(array.shape[1] == self.input_dim)
    feed_dict = {
      self.sy_input       : array
    }

    encoded_out, decoded_out = sess.run([self.encoded_out, self.decoded_out], feed_dict=feed_dict)
    return encoded_out, decoded_out

  def batch_encode(self, sess, array):
    """ Array is a N x M x input_dim and the output will be N x M x encoded_dim"""
    input_array = array.reshape((array.shape[0] * array.shape[1], array.shape[2]))
    encoded_out, decoded_out = self.encode(sess, input_array)
    return encoded_out.reshape((array.shape[0], array.shape[1], self.encoded_dim))
