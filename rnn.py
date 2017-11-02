"""
Class for creating a RNN using tensorflow. Translated from the a similar class written in lua
by chrispiech (https://github.com/chrispiech/DeepKnowledgeTracing/blob/master/scripts/rnn.lua)
"""
import tensorflow as tf
import numpy as np
from utils import *

class RNN(object):

  def __init__(self):
    pass
    self.hidden_dim
    self.num_questions
    self.input_dim = 2 * self.num_questions
    self.keep_prob


  def build(self):
    self.sy_input_memory     = tf.placeholder(shape=[None, self.hidden_dim], name="input_memory", dtype=tf.float32)
    self.sy_input_activity   = tf.placeholder(shape=[None, self.input_dim], name="input_activity", dtype=tf.float32)
    self.sy_next_question    = tf.placeholder(shape=[None, self.num_questions], name="input_activity", dtype=tf.float32)
    self.sy_question_correct = tf.placeholder(shape=[None], name="input_activity", dtype=tf.float32)

    # linM = tf.layers.dense(inputs=self.sy_input_memory, units=self.hidden_dim, activation=None, use_bias=False, name="linM")
    # linX = tf.layers.dense(inputs=self.sy_input_activity, units=self.hidden_dim, activation=None, use_bias=False, name="linX")
    # hidden = tf.tanh(linX + linM)
    # pred_input = tf.nn.dropout(hidden, self.keep_prob)
    #
    # linY = tf.layers.dense(inputs=pred_input, units=self.num_questions, activation=None, use_bias=False, name="linY")
    # pred = tf.reduce_sum(tf.sigmoid(linY) * self.sy_next_question, axis=1)
    # err = tf.losses.sigmoid_cross_entropy(self.sy_question_correct, linY, weights=self.sy_next_question)

    
