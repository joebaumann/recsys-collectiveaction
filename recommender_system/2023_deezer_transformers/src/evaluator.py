from math import log2, floor
import torch
import numpy as np
import pdb

class Evaluator():
  """ A class dedicated to computing metrics given a ground truth"""
  def __init__(self, data_manager, gt, n_recos = 500):
    self.gt = gt
    self.test_size = len(self.gt)
    self.n_recos = n_recos
    self.song_pop = torch.LongTensor(data_manager.song_pop)
    self.song_artist = data_manager.song_artist

  def get_test_set_playlists_to_consider(self, test_dataset, recos, test_set_constraint, targeted_base_songs):
    if test_set_constraint == "entire_test_set":
      # return 1 for all playlist in the test_dataset
      return np.ones(len(test_dataset), dtype=int)
    elif test_set_constraint == "targeted_base_songs_in_seed":
      targeted_base_songs_set = set(targeted_base_songs)
      # return 1 for all playlists in test_dataset that contain at least one targeted base song, else 0
      # use song-1 since testset dataloader returns songs with index starting at 1
      return np.array([1 if any(song-1 in targeted_base_songs_set for song in playlist) else 0 for playlist in test_dataset])
    elif test_set_constraint == "targeted_base_songs_in_recos":
      targeted_base_songs_set = set(targeted_base_songs)
      # return 1 for all song recommendations in recos that contain at least one targeted base song, else 0
      return np.array([
        1 if any(song in targeted_base_songs_set for song in recs[:len(gt_songs)]) else 0
        for recs, gt_songs in zip(recos, self.gt)
        ])

  def compute_all_recalls(self, recos):
    n = len(self.gt)
    return np.array([self.compute_single_recall(recos[i], self.gt[i]) for i in range(n)])

  def compute_single_RR(self, recos, gt):
    if len(gt) ==0:
      return 1
    return max([1/(i+1) for i in range(len(recos)) if recos[i] in gt] + [0])

  def compute_all_RRs(self, recos):
    n = len(self.gt)
    return np.array([self.compute_single_RR(recos[i], self.gt[i]) for i in range(n)])

  def compute_single_recall(self, recos, gt):
    R = len(gt)
    if R == 0:
      return 1
    return len(set(recos).intersection(gt)) / R

  def compute_single_precision(self, recos, gt):
    R = len(gt)
    if R == 0:
      return 1
    return len(set(recos).intersection(gt)) / len(recos)

  def compute_all_precisions(self, recos):
    n = len(self.gt)
    return np.array([self.compute_single_precision(recos[i], self.gt[i]) for i in range(n)])

  def compute_single_R_precision(self, recos, gt):
    R = len(gt)
    if R == 0:
      return 1
    song_score = len(set(recos[:R]).intersection(gt))
    artist_score = len(set(self.song_artist[recos.astype(np.int64)].astype(np.int64)).intersection(set(self.song_artist[list(gt)].astype(np.int64))))
    return (song_score + 0.25*artist_score) / R

  def compute_all_R_precisions(self, recos): #TODO : add artist share
    n = len(self.gt)
    return np.array([self.compute_single_R_precision(recos[i], self.gt[i]) for i in range(n)])

  def compute_single_signal_rec(self, recos, gt, signal):
    R = len(gt)
    if R == 0:
      return 0
    #pdb.set_trace()
    nr_of_signal_recs = len([r for r in recos if int(r) == int(signal)])
    # Count the number of times the signal appears
    signal_count = np.count_nonzero(recos == signal)
    if nr_of_signal_recs != signal_count:
      pdb.set_trace()
    assert nr_of_signal_recs == signal_count, "nr_of_signal_recs: %d, signal_count: %d" % (nr_of_signal_recs, signal_count)
    return signal_count

  def compute_all_signal_recs(self, recos, signal):
    n = len(self.gt)
    return np.array([self.compute_single_signal_rec(recos[i], self.gt[i], signal) for i in range(n)])

  def compute_single_signal_rec_in_holdouts(self, recos, gt, signal):
    R = len(gt)
    if R == 0:
      return 0
    # Get the first R recommendations
    top_R_recos = recos[:R]
    nr_of_signal_recs = len([r for r in top_R_recos if int(r) == int(signal)])
    # Count the number of times the signal appears in the top R recommendations
    signal_count = np.count_nonzero(top_R_recos == signal)
    if nr_of_signal_recs != signal_count:
      pdb.set_trace()
    assert nr_of_signal_recs == signal_count, "nr_of_signal_recs in holdouts: %d, signal_count in holdouts: %d" % (nr_of_signal_recs, signal_count)
    return signal_count

  def compute_all_signal_recs_in_holdouts(self, recos, signal):
    n = len(self.gt)
    return np.array([self.compute_single_signal_rec_in_holdouts(recos[i], self.gt[i], signal) for i in range(n)])

  def compute_all_seed_song_recs_in_holdouts(self, recos, seed_songs):
    n = len(self.gt)
    total_recs = np.zeros(n)  # Initialize an array of zeros with the same length as self.gt
    for seed_song in seed_songs:
      # Compute recommendations for each seed song and add it to the total
      total_recs += np.array([self.compute_single_signal_rec_in_holdouts(recos[i], self.gt[i], seed_song) for i in range(n)])
    return total_recs

  def compute_single_signal_rank(self, recos, gt, signal):
    R = len(gt)
    # Find the index (rank) of the first occurrence of the signal
    indices = np.where(recos == signal)[0]
    if indices.size > 0:
        signal_rank = indices[0] + 1  # Adding 1 because index is 0-based
    else:
      signal_rank = np.nan  # Signal not found in the recommendations
    return signal_rank
  
  def compute_all_signal_ranks(self, recos, signal):
    n = len(self.gt)
    ranks = np.array([self.compute_single_signal_rank(recos[i], self.gt[i], signal) for i in range(n)])
    return ranks

  def compute_single_signal_rank_in_holdouts(self, recos, gt, signal):
    R = len(gt)
    # Get the first R recommendations
    top_R_recos = recos[:R]
    # Find the index (rank) of the first occurrence of the signal
    indices = np.where(top_R_recos == signal)[0]
    if indices.size > 0:
        signal_rank = indices[0] + 1  # Adding 1 because index is 0-based
    else:
      signal_rank = np.nan  # Signal not found in the recommendations
    return signal_rank
  
  def compute_all_signal_ranks_in_holdouts(self, recos, signal):
    n = len(self.gt)
    ranks = np.array([self.compute_single_signal_rank_in_holdouts(recos[i], self.gt[i], signal) for i in range(n)])
    return ranks

  def compute_single_signal_click(self, recos, signal):
      n_recos = recos.shape[0]
      return next((floor(i/10) for i in range(n_recos) if recos[i] == signal), 51)

  def compute_all_signal_clicks(self, recos, signal):
      n = len(self.gt)
      return np.array([self.compute_single_signal_click(recos[i], signal) for i in range(n)])

  def compute_single_click(self, recos, gt):
    n_recos = recos.shape[0]
    if len(gt) ==0:
      return 1
    return next((floor(i/10) for i in range(n_recos) if recos[i] in gt), 51)

  def compute_all_clicks(self, recos):
    n = len(self.gt)
    return np.array([self.compute_single_click(recos[i], self.gt[i]) for i in range(n)])

  def compute_pop(self, recos):
    s = recos.shape
    pop = torch.gather(self.song_pop, 0, torch.LongTensor(recos.reshape(-1))).reshape(s)
    return pop

  def compute_norm_pop(self, recos):
    pop = self.compute_pop(recos)
    max_pop = self.song_pop.max()
    min_pop = self.song_pop.min()
    return np.array((pop - min_pop) / (max_pop -min_pop)).mean(axis=1)

  def compute_cov(self, recos):
    return len(np.unique(recos)) / len(self.song_pop)

  def compute_single_dcg(self, recos, gt):
    if len(gt) ==0:
      return 1
    return sum([1/log2(i+2) for i in range(len(recos)) if recos[i] in gt])

  def compute_single_ndcg(self, recos, gt):
    dcg = self.compute_single_dcg(recos, gt)
    idcg = self.compute_single_dcg(list(gt), gt)
    return dcg/idcg
  
  def compute_all_ndcgs(self, recos):
    n = len(self.gt)
    return np.array([self.compute_single_ndcg(recos[i], self.gt[i]) for i in range(n)])