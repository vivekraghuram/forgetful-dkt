import numpy as np
import pandas as pd

class Data(object):

  def training_batches(self, batch_size):
    """ Returns an iterator over training data with given batch size and
        format (inputs, targets, target_masks) """
    raise NotImplementedError

  def testing_batches(self, batch_size):
    """ Returns an iterator over testing data with given batch size and
        format (inputs, targets, target_masks) """
    raise NotImplementedError

  def shuffle(self):
    """ Shuffles the training data """
    raise NotImplementedError

  @property
  def num_sequences(self):
    raise NotImplementedError

  @property
  def max_sequence_length(self):
    raise NotImplementedError

class SyntheticData(Data):

  def __init__(self, path="data/synthetic_c2_q50.csv"):
    self.path = path
    self.data = np.genfromtxt(path, delimiter=',')
    self.data = self.data.reshape(self.data.shape[0], self.data.shape[1], 1)

    self.num_students  = self.data.shape[0]
    self.num_questions = self.data.shape[1]

    self.inputs  = self.data[:, :-1]
    self.targets = self.data[:, 1:]
    self.target_masks = np.ones(self.targets.shape)

    self.training_cutoff = int(0.8 * self.num_students)

  def training_batches(self, batch_size):
    start_idx = 0
    end_idx = batch_size
    while end_idx < self.training_cutoff:
      yield self.inputs[start_idx:end_idx], self.targets[start_idx:end_idx], self.target_masks[start_idx:end_idx]
      start_idx, end_idx = end_idx, end_idx + batch_size

  def testing_batches(self, batch_size):
    start_idx = self.training_cutoff
    end_idx = start_idx + batch_size
    while end_idx < self.num_students:
      yield self.inputs[start_idx:end_idx], self.targets[start_idx:end_idx], self.target_masks[start_idx:end_idx]
      start_idx, end_idx = end_idx, end_idx + batch_size

  def shuffle(self):
    indices = np.arange(self.training_cutoff)
    np.random.shuffle(indices)
    self.data[0:self.training_cutoff] = self.data[indices]
