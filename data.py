import numpy as np
import pandas as pd

class Data(object):

  def k_fold(self, k):
    """ Returns an iterator over range(1, k+1), rotating the available training
    and testing data with each iteration, such that a different fraction 1/k
    of the data will be the testing data each iteration """
    raise NotImplementedError

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

  def k_fold(self, k):
    self.training_cutoff = int((k - 1.0) / k * self.num_students)
    for i in range(k):
      np.roll(self.inputs, self.num_students - self.training_cutoff, axis=0)
      np.roll(self.targets, self.num_students - self.training_cutoff, axis=0)
      np.roll(self.target_masks, self.num_students - self.training_cutoff, axis=0)
      yield i + 1

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
    self.inputs[0:self.training_cutoff] = self.inputs[indices]
    self.targets[0:self.training_cutoff] = self.targets[indices]
    self.target_masks[0:self.training_cutoff] = self.target_masks[indices]

class SingleExerciseData(SyntheticData):

  def __init__(self, seq_path, mask_path):
    self.seq_path = seq_path
    self.mask_path = mask_path
    self.data = np.load(self.seq_path)
    self.target_masks = np.load(self.mask_path)[:, 1:]
    self.target_masks = self.target_masks.reshape(self.target_masks.shape[0], self.target_masks.shape[1], 1)

    self.num_students  = self.data.shape[0]
    self.num_questions = self.data.shape[1]

    self.inputs = self.data[:, :-1]
    self.targets = self.data[:, 1:, 0:1] # we only need to predict correctness

    self.training_cutoff = int(0.8 * self.num_students)

class LoadedExerciseData(SyntheticData):

  def __init__(self, seq_path, mask_path, corrects_path):
    self.seq_path = seq_path
    self.mask_path = mask_path
    self.corrects_path = corrects_path
    self.data = np.load(self.seq_path)
    self.target_masks = np.load(self.mask_path)[:, 1:]
    self.targetss = np.load(self.mask_path)[:, 1:]

    self.num_students  = self.data.shape[0]
    self.num_questions = self.data.shape[1]

    self.inputs = self.data[:, :-1]

    self.training_cutoff = int(0.8 * self.num_students)


class InMemoryExerciseData(SyntheticData):

  def __init__(self, sequences, mask, corrects):
    self.data = sequences
    self.target_masks = mask[:, 1:]
    self.targets = corrects[:, 1:]

    self.num_students  = self.data.shape[0]
    self.num_questions = self.data.shape[1]

    self.inputs = self.data[:, :-1]

    self.training_cutoff = int(0.8 * self.num_students)
