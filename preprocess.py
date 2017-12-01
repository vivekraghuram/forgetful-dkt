import numpy as np
import pandas as pd
from forgetful_dkt.data import InMemoryExerciseData

class DataProcessor(object):
  """
  Class to load and preprocess data.

  Usage:
  dp = DataProcessor('vivek', ['exponents_1'])
  data = dp.correctness_only().get_data()
  ....
  dp = DataProcessor('vivek', ['exponents_1'])
  seq_file_name, mask_file_name, corrects_file_name = dp.correctness_only().save_data('filename')
  data = LoadedExerciseData(seq_file_name, mask_file_name, corrects_file_name)
  """

  def __init__(self,
               user_name,
               input_file_names,
               min_seq_length=5,
               max_seq_length=500,
               min_correct=1):
    """
    Load data into DataProcessor, remove NaNs and violations of length/correctness constraints
    user_name: string, used to avoid file name conflicts when saving data
    input_file_names: list<string>, names of khan academy exercises stored in /deepedu/research/dktforgetting/data/
    min_seq_length: int, the minimum number of responses by a specific user
    max_seq_length: int, the maximum number of responses by a specific user
    min_correct: int, the minimum number of correct responses by a specific user
    """

    self.is_data_processed = False
    self.processed_vectors = {}
    self.data_key = None
    self.user_name = user_name

    self._input_file_names = input_file_names
    self._min_seq_length = min_seq_length
    self._max_seq_length = max_seq_length
    self._min_correct = min_correct

    self.names = pd.read_csv('/deepedu/research/dktforgetting/header.csv', header=None)

    dfs = []
    for file_name in input_file_names:
        df = pd.DataFrame({'uid':[], 'date':[], 'exercise':[], 'correct':[]})
        dfs.append(self.get_dfs(file_name, df).dropna(axis=0, how='any'))

    dfs = pd.concat(dfs)
    dfs['count'] = 1
    grouped_dfs = dfs.groupby('uid').sum()
    self.filtered_uids = grouped_dfs[(grouped_dfs['count'] >= min_seq_length) &
                                     (grouped_dfs['count'] <= max_seq_length) &
                                     (grouped_dfs['correct'] >= min_correct)].reset_index()['uid']
    self.raw_data = dfs[(dfs['uid'].isin(self.filtered_uids))].drop(['count'],
                                                                    axis=1).sort_values('date')
    self.exercise_names = self.raw_data.exercise.unique()

  def get_dfs(self, exercise_name, exercise_df):
    """
    Get a data frame for a specific exercise
    exercise_name: string, specifying exercise name
    exercise_df: pandas dataframe, a dataframe in which to store the exercise data
    """
    exercise_file = '/deepedu/research/dktforgetting/data/' + exercise_name +'.csv'
    df = pd.read_csv(exercise_file, header = None, names = self.names.iloc[0],
                     usecols = ['f0_', 'time_done', 'exercise', 'correct'])
    df['uid'] = pd.to_numeric(df['f0_'].str.split('-').str[1], errors='coerce')
    df['date'] = pd.to_datetime(df['time_done']).astype(np.int64) // 10**9 # get unix time
    df.drop(['f0_', 'time_done'], axis=1, inplace=True)
    return exercise_df.append(df)

  def correctness_only(self, verbose=True):
    """
    Processes data to only include correctness of the previous question as input.
    Ex. For exercises A and B:
        [0, 0, 1, 0]
        Means that the current response is for exercise B and is incorrect.
    """
    if 'correctness_only_sequences' in self.processed_vectors:
      self.data_key = 'correctness_only'
      return self

    corrects = np.zeros((len(self.filtered_uids), self._max_seq_length, len(self.exercise_names)))
    mask = np.zeros((len(self.filtered_uids), self._max_seq_length, len(self.exercise_names)))
    sequences = np.zeros((len(self.filtered_uids),
                          self._max_seq_length,
                          2 * len(self.exercise_names)))

    exercise_index_map = {}
    for idx, exercise in enumerate(self.exercise_names):
      exercise_index_map[exercise] = idx * 2

    for row_idx, uid in enumerate(self.filtered_uids):
      uid_df = self.raw_data[self.raw_data['uid'] == uid]
      col_idx = 0

      for _, event_dict in uid_df.iterrows():
          idx = exercise_index_map[event_dict['exercise']]
          sequences[row_idx, col_idx, idx] = 1
          sequences[row_idx, col_idx, idx + 1] = event_dict['correct']

          corrects[row_idx, col_idx, idx // 2] = event_dict['correct']
          mask[row_idx, col_idx, idx // 2] = 1
          col_idx += 1

      if verbose and row_idx % 100 == 0:
          print("Processed %d" % row_idx)

    self.processed_vectors['correctness_only_sequences'] = sequences
    self.processed_vectors['correctness_only_mask'] = mask
    self.processed_vectors['correctness_only_corrects'] = corrects
    self.data_key = 'correctness_only'
    return self

  def continuous_delay_by_exercise(self, verbose=True):
    """
    Processes data to include correctness and delay in minutes with delay split by exercise.
    Ex. For exercises A and B:
        [0, 0, 15, 0, 1, 0, 2, 0]
        Means that it has been 15 minutes since the user answered and exercise A question and
        2 minutes since they last answered an exercise B question. The current response is for
        exercise B and is incorrect. The user has previously answer questions for both exercises.

        [0, 0, 15, 0, 1, 0, 0, 1]
        Means that it has been 15 minutes since the user answered and exercise A question and
        the user has not answered an exercise B question before. The current response is for
        exercise B and is incorrect.
    """
    if 'continuous_delay_by_exercise_sequences' in self.processed_vectors:
      self.data_key = 'continuous_delay_by_exercise'
      return self

    corrects = np.zeros((len(self.filtered_uids), self._max_seq_length, len(self.exercise_names)))
    mask = np.zeros((len(self.filtered_uids), self._max_seq_length, len(self.exercise_names)))
    sequences = np.zeros((len(self.filtered_uids),
                          self._max_seq_length,
                          4 * len(self.exercise_names)))

    exercise_index_map = {}
    for idx, exercise in enumerate(self.exercise_names):
      exercise_index_map[exercise] = idx * 4

    row_idx = 0
    for uid in self.filtered_uids:
      uid_df = self.raw_data[self.raw_data['uid'] == uid]
      col_idx = 0
      prev_evdicts = [None] * len(self.exercise_names)

      for _, event_dict in uid_df.iterrows():
        idx = exercise_index_map[event_dict['exercise']]
        sequences[row_idx, col_idx, idx] = 1
        sequences[row_idx, col_idx, idx + 1] = event_dict['correct']

        if col_idx > 0:
          for i in range(4 * len(self.exercise_names)):
            if i % 4 == 2:
              sequences[row_idx, col_idx, i] = sequences[row_idx, col_idx - 1, i]

        if prev_evdicts[idx] is None:
          sequences[row_idx, col_idx, idx + 3] = 1
        else:
          sequences[row_idx, col_idx, idx + 2] = (event_dict['date'] - prev_evdicts[idx]['date']).total_seconds() / 60

        prev_evdicts[idx] = event_dict
        corrects[row_idx, col_idx, idx // 4] = event_dict['correct']
        mask[row_idx, col_idx, idx // 4] = 1
        col_idx += 1

      row_idx += 1
      if verbose and row_idx % 100 == 0:
          print("Processed %d" % row_idx)

    self.processed_vectors['continuous_delay_by_exercise_sequences'] = sequences
    self.processed_vectors['continuous_delay_by_exercise_mask'] = mask
    self.processed_vectors['continuous_delay_by_exercise_corrects'] = corrects
    self.data_key = 'continuous_delay_by_exercise'
    return self

  def continuous_delay_aggregate(self, verbose=True):
    """
    Processes data to include correctness and delay in minutes with delay being since any response.
    Ex. For exercises A and B:
        [0, 0, 2, 0, 1, 0, 2, 0]
        Means that it has been 2 minutes since the user last answered any questions. The current
        response is for exercise B and is incorrect. The user has previously answered a question.
    """
    if 'continuous_delay_aggregate_sequences' in self.processed_vectors:
      self.data_key = 'continuous_delay_aggregate'
      return self

    corrects = np.zeros((len(self.filtered_uids), self._max_seq_length, len(self.exercise_names)))
    mask = np.zeros((len(self.filtered_uids), self._max_seq_length, len(self.exercise_names)))
    sequences = np.zeros((len(self.filtered_uids),
                          self._max_seq_length,
                          4 * len(self.exercise_names)))

    exercise_index_map = {}
    for idx, exercise in enumerate(self.exercise_names):
      exercise_index_map[exercise] = idx * 4

    row_idx = 0
    for uid in self.filtered_uids:
      uid_df = self.raw_data[self.raw_data['uid'] == uid]
      col_idx = 0
      prev_event_dict = None

      for _, event_dict in uid_df.iterrows():
        idx = exercise_index_map[event_dict['exercise']]
        sequences[row_idx, col_idx, idx] = 1
        sequences[row_idx, col_idx, idx + 1] = event_dict['correct']
        if prev_event_dict is None:
          for i in range(4 * len(self.exercise_names)):
            if i % 4 == 3:
              sequences[row_idx, col_idx, i] = 1
        else:
          time_difference = (event_dict['date'] - prev_event_dict['date']) / 60
          for i in range(4 * len(self.exercise_names)):
            if i % 4 == 2:
              sequences[row_idx, col_idx, i] = time_difference

        prev_event_dict = event_dict
        corrects[row_idx, col_idx, idx // 4] = event_dict['correct']
        mask[row_idx, col_idx, idx // 4] = 1
        col_idx += 1

      row_idx += 1
      if verbose and row_idx % 100 == 0:
          print("Processed %d" % row_idx)

    self.processed_vectors['continuous_delay_aggregate_sequences'] = sequences
    self.processed_vectors['continuous_delay_aggregate_mask'] = mask
    self.processed_vectors['continuous_delay_aggregate_corrects'] = corrects
    self.data_key = 'continuous_delay_aggregate'
    return self

  def bucketed_delay_by_exercise(self, buckets, immediate_buckets=None, remove_immediate_only_seqs=False, verbose=True):
    """
    Processes data to include correctness and bucketed delay with delay split by exercise.
    Ex. For exercises A, B and buckets [5, 24, None]:
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                  ^^^^^^^           ^^^^^^^  ^^^^^^^           ^^^^^^^
                  A <5hrs           A >24hrs B no prev.        B <24hrs
        Means that it has been less than 5 hours since the user last responded to A and has no
        previous responses to B. The current response is to B and it is incorrect.

    buckets: list<float>, Specifies the window in hours for the bucket. 'None' specifies the default
                          bucket.
    immediate_buckets: list<float>, A subset of buckets. Specifies those buckets to treat as
                                    immediate buckets for filtering.
    remove_immediate_only_seqs: boolean, Whether to remove sequences that only have immediate
                                         buckets
    """
    if 'bucketed_delay_by_exercise_sequences' in self.processed_vectors:
      self.data_key = 'bucketed_delay_by_exercise'
      return self

    num_buckets = 1 + len(buckets)
    corrects = np.zeros((len(self.filtered_uids), self._max_seq_length, len(self.exercise_names)))
    mask = np.zeros((len(self.filtered_uids), self._max_seq_length, len(self.exercise_names)))
    sequences = np.zeros((len(self.filtered_uids),
                          self._max_seq_length,
                          3 * len(self.exercise_names) * num_buckets))

    exercise_index_map = {}
    for idx, exercise in enumerate(self.exercise_names):
      exercise_index_map[exercise] = idx

    row_idx = 0
    not_immediate_only_rows = []
    for uid in self.filtered_uids:
      uid_df = self.raw_data[self.raw_data['uid'] == uid]
      col_idx = 0
      prev_evdicts = [None] * len(self.exercise_names)
      immediate_only = True

      for _, event_dict in uid_df.iterrows():
        exercise_idx = exercise_index_map[event_dict['exercise']]
        bucket_idx = self.bucket_idx(buckets, event_dict, prev_evdicts[exercise_idx])
        idx = exercise_idx * 3 * num_buckets + 3 * bucket_idx
        sequences[row_idx, col_idx, idx] = 1
        sequences[row_idx, col_idx, idx + 1] = event_dict['correct']
        sequences[row_idx, col_idx, idx + 2] = 1

        if col_idx > 0:
          for i in range(len(self.exercise_names)):
            if i != exercise_idx:
              for j in range(num_buckets):
                sequences[row_idx, col_idx, i * 3 * num_buckets + 3 * j + 2] = sequences[row_idx, col_idx - 1, i * 3 * num_buckets + 3 * j + 2]

        if immediate_buckets is not None:
          if bucket_idx > 0 and buckets[bucket_idx - 1] not in immediate_buckets:
            immediate_only = False

        prev_evdicts[exercise_idx] = event_dict
        corrects[row_idx, col_idx, exercise_idx] = event_dict['correct']
        mask[row_idx, col_idx, exercise_idx] = 1
        col_idx += 1

      if not immediate_only:
        not_immediate_only_rows.append(row_idx)
      row_idx += 1
      if verbose and row_idx % 100 == 0:
          print("Processed %d" % row_idx)

    if remove_immediate_only_seqs:
      sequences = sequences[not_immediate_only_rows]
      mask = mask[not_immediate_only_rows]
      corrects = corrects[not_immediate_only_rows]

    self.processed_vectors['bucketed_delay_by_exercise_sequences'] = sequences
    self.processed_vectors['bucketed_delay_by_exercise_mask'] = mask
    self.processed_vectors['bucketed_delay_by_exercise_corrects'] = corrects
    self.data_key = 'bucketed_delay_by_exercise'
    return self

  def bucketed_delay_aggregate(self, buckets, immediate_buckets=None, remove_immediate_only_seqs=False, verbose=True):
    """
    Processes data to include correctness and bucketed delay with delay being since any response.
    Ex. For exercises A, B and buckets [5, 24, None]:
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
                  ^^^^^^^           ^^^^^^^  ^^^^^^^           ^^^^^^^
                  A <5hrs           A >24hrs B no prev.        B <24hrs
        Means that it has been less than 5 hours since the user last responded to any question.
        The current response is to B and it is incorrect.

    buckets: list<float>, Specifies the window in hours for the bucket. 'None' specifies the default
                          bucket.
    immediate_buckets: list<float>, A subset of buckets. Specifies those buckets to treat as
                                    immediate buckets for filtering.
    remove_immediate_only_seqs: boolean, Whether to remove sequences that only have immediate
                                         buckets
    """
    if 'bucketed_delay_aggregate_sequences' in self.processed_vectors:
      self.data_key = 'bucketed_delay_aggregate'
      return self

    num_buckets = 1 + len(buckets)
    corrects = np.zeros((len(self.filtered_uids), self._max_seq_length, len(self.exercise_names)))
    mask = np.zeros((len(self.filtered_uids), self._max_seq_length, len(self.exercise_names)))
    sequences = np.zeros((len(self.filtered_uids),
                          self._max_seq_length,
                          3 * len(self.exercise_names) * num_buckets))

    exercise_index_map = {}
    for idx, exercise in enumerate(self.exercise_names):
      exercise_index_map[exercise] = idx

    row_idx = 0
    not_immediate_only_rows = []
    for uid in self.filtered_uids:
      uid_df = self.raw_data[self.raw_data['uid'] == uid]
      col_idx = 0
      prev_event_dict = None
      immediate_only = True

      for _, event_dict in uid_df.iterrows():
        exercise_idx = exercise_index_map[event_dict['exercise']]
        bucket_idx = self.bucket_idx(buckets, event_dict, prev_event_dict)
        idx = exercise_idx * 3 * num_buckets + 3 * bucket_idx

        sequences[row_idx, col_idx, idx] = 1
        sequences[row_idx, col_idx, idx + 1] = event_dict['correct']
        for i in range(len(self.exercise_names)):
          sequences[row_idx, col_idx, i * 3 * num_buckets + 3 * bucket_idx + 2] = 1

        if immediate_buckets is not None:
          if bucket_idx > 0 and buckets[bucket_idx - 1] not in immediate_buckets:
            immediate_only = False

        prev_event_dict = event_dict
        corrects[row_idx, col_idx, exercise_idx] = event_dict['correct']
        mask[row_idx, col_idx, exercise_idx] = 1
        col_idx += 1

      if not immediate_only:
        not_immediate_only_rows.append(row_idx)
      row_idx += 1
      if verbose and row_idx % 100 == 0:
          print("Processed %d" % row_idx)

    if remove_immediate_only_seqs:
      sequences = sequences[not_immediate_only_rows]
      mask = mask[not_immediate_only_rows]
      corrects = corrects[not_immediate_only_rows]

    self.processed_vectors['bucketed_delay_aggregate_sequences'] = sequences
    self.processed_vectors['bucketed_delay_aggregate_mask'] = mask
    self.processed_vectors['bucketed_delay_aggregate_corrects'] = corrects
    self.data_key = 'bucketed_delay_aggregate'
    return self

  def bucket_idx(self, buckets, event_dict, prev_event_dict):
    if prev_event_dict is None:
      return 0

    time_difference_hrs = (event_dict['date'] - prev_event_dict['date']) / 3600
    for idx, bucket in enumerate(buckets):
      if bucket == None:
        return idx + 1
      if time_difference_hrs < bucket:
        return idx + 1
    raise Exception("No compatible buckets: {}".format(buckets))

  def get_data(self):
    if self.data_key is None:
      raise Exception("No data has been processed yet. See source file for usage details.")
    sequences = self.processed_vectors['%s_sequences' % self.data_key]
    mask = self.processed_vectors['%s_mask' % self.data_key]
    corrects = self.processed_vectors['%s_corrects' % self.data_key]
    return InMemoryExerciseData(sequences, mask, corrects)

  def save_data(self, output_file_name):
    if self.data_key is None:
      raise Exception("No data has been processed yet. See source file for usage details.")
    sequences = self.processed_vectors['%s_sequences' % self.data_key]
    mask = self.processed_vectors['%s_mask' % self.data_key]
    corrects = self.processed_vectors['%s_corrects' % self.data_key]

    seq_file_name = "../pp-data/%s/%s_seq_%d.npy" % (self.user_name, output_file_name)
    mask_file_name = "../pp-data/%s/%s_mask_%d.npy" % (self.user_name, output_file_name)
    corrects_file_name = "../pp-data/%s/%s_corrects_%d.npy" % (self.user_name, output_file_name)

    np.save(seq_file_name, sequences)
    np.save(mask_file_name, mask)
    np.save(corrects_file_name, corrects)

    return seq_file_name, mask_file_name, corrects_file_name
