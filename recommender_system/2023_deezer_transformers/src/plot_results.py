import numpy as np
import argparse
from src.data_manager.data_manager import DataManager, EvaluationDataset
import seaborn as sns
import matplotlib.pyplot as plt
import os
import tqdm
from matplotlib.ticker import FormatStrFormatter
from pathlib import Path
import yaml
from munch import munchify
import csv
import pdb
import sys
from utils import get_nr_of_unique_tracks, get_target_song_info, get_onefamoussong, get_severalfamoussongs, disable_progress_bar
from collections import Counter
import pandas as pd
import copy
import utils

def confidence_interval(metrics):
    # Compute 95% confidence interval
    n = metrics.shape[0]
    std = metrics.std()
    return 1.96 * (std/np.sqrt(n))

def create_grouping_matrix():
    # Multiplying by this matrix gives a grouped average over 1000 rows.
    # Useful for averaging over test playlists with the same n_seed
    M = np.zeros((10000, 10))
    kernel = np.ones((1,1000))/ 1000
    for i in range(10):
      M[1000*i: 1000* (i+1), i] = kernel
    return M

@utils.timer
def calculate_results(rec, epoch, test_evaluator, data_manager, outfile_path, args, signal_as_track_ids=None, signal_by_track_id=None, targeted_base_song_as_track_ids=None, baseline_name="", baseline_recs=None):

    results = []

    M = create_grouping_matrix()
    precs = test_evaluator.compute_all_precisions(rec).dot(M) # * 100 # for %
    recalls = test_evaluator.compute_all_recalls(rec).dot(M) # * 100 # for %
    R_precs = test_evaluator.compute_all_R_precisions(rec).dot(M) # * 100 # for %
    ndcgs = test_evaluator.compute_all_ndcgs(rec).dot(M) # * 100 # for %
    clicks = test_evaluator.compute_all_clicks(rec).dot(M)
    if baseline_name=="":
        # TODO: does not work for baselines, since we add song -1 to simulate the wors case(s)
        norm_pop = test_evaluator.compute_norm_pop(rec).dot(M) # * 100 # for %

    #pdb.set_trace()
    ground_truths = test_evaluator.gt
    ground_truth_size = np.array([len(s) for s in ground_truths])
    ground_truth_size_1000 = ground_truth_size.reshape(-1, 1000).sum(axis=1)

    test_set_constraints = {'entire_test_set': None}
    if targeted_base_song_as_track_ids and len(targeted_base_song_as_track_ids) > 0:
        test_set_constraints['targeted_base_songs_in_seed'] = targeted_base_song_as_track_ids
        test_set_constraints['targeted_base_songs_in_recos'] = targeted_base_song_as_track_ids
        nr_of_targeted_seed_songs = len(targeted_base_song_as_track_ids)
    else:
        nr_of_targeted_seed_songs = 0

        # TODO: get nr of base song recommendations without collective action
    
    test_dataset = EvaluationDataset(data_manager, data_manager.test_indices)
    
    if baseline_name=="":
        # collective action success
        #pdb.set_trace()
        if signal_as_track_ids is None:
            return [{
                "target_song": "-",
                "epoch": epoch,
                "is_baseline": baseline_name,
                "scenario": "all",
                "avg_r_precision": np.mean(R_precs),
                "avg_ndcg": np.mean(ndcgs),
                "avg_recommended_song_clicks": np.mean(clicks),
                "total_signal_recommendations": 0,
                "total_signal_recommendations_in_holdouts": 0,
                "avg_rank_signal": 0,
                "avg_rank_signal_in_holdouts": 0,
                "avg_signal_song_clicks": 0,
                "nr_of_test_set_song_recommendations_by_scenario": sum(ground_truth_size_1000),
                "test_set_constraint": None,
                "nr_of_considered_test_set_playlists": None,
                "nr_of_targeted_seed_songs": 0,
                "seed_song_recs_in_holdouts_1000": None
                #"avg_precision": np.mean(precs),
                #"avg_recall": np.mean(recalls),
                #"avg_recall": np.mean(recalls),
                #"popularity": np.mean(norm_pop),
            }]
            


        for test_set_constraint, targeted_base_songs in test_set_constraints.items():

            print("test_set_constraint:", test_set_constraint)
            print("targeted_base_songs:", targeted_base_songs)

            #if test_set_constraint == "entire_test_set" and targeted_base_song_as_track_ids and len(targeted_base_song_as_track_ids) > 0:
            if targeted_base_song_as_track_ids and len(targeted_base_song_as_track_ids) > 0:
                seed_song_recs_in_holdouts = test_evaluator.compute_all_seed_song_recs_in_holdouts(rec, targeted_base_song_as_track_ids)
                seed_song_recs_in_holdouts_1000 = seed_song_recs_in_holdouts.reshape(-1, 1000).sum(axis=1)
            else:
                seed_song_recs_in_holdouts = None
                seed_song_recs_in_holdouts_1000 = None

            # get the number of playlists (from the 10K total test set playlists) that are considered for the test set constraint
            # use the seed song recos WITHOUT COLLECTIVE ACTION to get those test time playlist where any of the seed songs would have been recommended
            if baseline_recs:
                # this is baseline_recs to check this based on seed song recomendations without collective action
                baseline_rec, targeted_base_song_as_track_ids_without_collective_action, test_dataset_without_collective_action = baseline_recs
                test_set_playlists_to_consider = test_evaluator.get_test_set_playlists_to_consider(test_dataset_without_collective_action, baseline_rec, test_set_constraint, targeted_base_song_as_track_ids_without_collective_action)
            else:
                test_set_playlists_to_consider = test_evaluator.get_test_set_playlists_to_consider(test_dataset, rec, test_set_constraint, targeted_base_songs)

            for signal_as_track_id in signal_as_track_ids:

                target_song = signal_by_track_id[signal_as_track_id]

                signal_recs = test_evaluator.compute_all_signal_recs(rec, signal_as_track_id)
                signal_recs_1000 = signal_recs.reshape(-1, 1000).sum(axis=1)
                signal_recs_in_holdouts = test_evaluator.compute_all_signal_recs_in_holdouts(rec, signal_as_track_id) * test_set_playlists_to_consider
                signal_recs_in_holdouts_1000 = signal_recs_in_holdouts.reshape(-1, 1000).sum(axis=1)
                signal_ranks = test_evaluator.compute_all_signal_ranks(rec, signal_as_track_id) * test_set_playlists_to_consider
                signal_ranks_1000 = np.nanmean(signal_ranks.reshape(-1, 1000), axis=1) # np.nan values are ignored
                signal_ranks_in_holdouts = test_evaluator.compute_all_signal_ranks_in_holdouts(rec, signal_as_track_id) * test_set_playlists_to_consider
                signal_ranks_in_holdouts_1000 = np.nanmean(signal_ranks_in_holdouts.reshape(-1, 1000), axis=1) # np.nan values are ignored
                signal_clicks_1000 = test_evaluator.compute_all_signal_clicks(rec, signal_as_track_id).dot(M)
                
                # Calculate average values for overall results
                scenario_all_results = {
                    "target_song": target_song,
                    "epoch": epoch,
                    "is_baseline": False,
                    "scenario": "all",
                    "avg_r_precision": np.mean(R_precs),
                    "avg_ndcg": np.mean(ndcgs),
                    "avg_recommended_song_clicks": np.mean(clicks),
                    "total_signal_recommendations": sum(signal_recs_1000),
                    "total_signal_recommendations_in_holdouts": sum(signal_recs_in_holdouts_1000),
                    "avg_rank_signal": np.mean(signal_ranks_1000),
                    "avg_rank_signal_in_holdouts": np.mean(signal_ranks_in_holdouts_1000),
                    "avg_signal_song_clicks": np.mean(signal_clicks_1000),
                    "nr_of_test_set_song_recommendations_by_scenario": sum(ground_truth_size_1000),
                    "test_set_constraint": test_set_constraint,
                    "nr_of_considered_test_set_playlists": sum(test_set_playlists_to_consider),
                    "nr_of_targeted_seed_songs": nr_of_targeted_seed_songs,
                    "seed_song_recs_in_holdouts_1000": sum(seed_song_recs_in_holdouts_1000) if seed_song_recs_in_holdouts_1000 is not None else None
                    #"avg_precision": np.mean(precs),
                    #"avg_recall": np.mean(recalls),
                    #"avg_recall": np.mean(recalls),
                    #"popularity": np.mean(norm_pop),
                }
                results.append(scenario_all_results)
                
                if seed_song_recs_in_holdouts_1000 is not None:
                    results_to_zip = [data_manager.N_SEED_SONGS, R_precs, ndcgs, clicks, signal_recs_1000, signal_recs_in_holdouts_1000, signal_ranks_1000, signal_ranks_in_holdouts_1000, signal_clicks_1000, ground_truth_size_1000, seed_song_recs_in_holdouts_1000]
                else:
                    results_to_zip = [data_manager.N_SEED_SONGS, R_precs, ndcgs, clicks, signal_recs_1000, signal_recs_in_holdouts_1000, signal_ranks_1000, signal_ranks_in_holdouts_1000, signal_clicks_1000, ground_truth_size_1000, [None for _ in data_manager.N_SEED_SONGS]]

                # Assert that all elements have the same length
                for index, e in enumerate(results_to_zip):
                    assert len(e) == len(results_to_zip[0]), f"Result element at position {index} in the 'results_to_zip' list does not have the same size. Its size is {len(e)} but it should be {len(results_to_zip[0])}."


                # Calculate average values for each scenario
                for n_seed, n_seed_R_precs, n_seed_ndcgs, n_seed_clicks, n_seed_signal_recs_1000, n_seed_signal_recs_in_holdouts_1000, n_seed_signal_ranks_1000, n_seed_signal_ranks_in_holdouts_1000, n_seed_signal_clicks_1000, n_seed_ground_truth_size_1000, n_seed_seed_song_recs_in_holdouts_1000 in zip(*results_to_zip):

                    combined_scenario_results = {
                        "target_song": target_song,
                        "epoch": epoch,
                        "is_baseline": False,
                        "scenario": str(n_seed),
                        "avg_r_precision": n_seed_R_precs,
                        "avg_ndcg": n_seed_ndcgs,
                        "avg_recommended_song_clicks": n_seed_clicks,
                        "total_signal_recommendations": n_seed_signal_recs_1000,
                        "total_signal_recommendations_in_holdouts": n_seed_signal_recs_in_holdouts_1000,
                        "avg_rank_signal": n_seed_signal_ranks_1000,
                        "avg_rank_signal_in_holdouts": n_seed_signal_ranks_in_holdouts_1000,
                        "avg_signal_song_clicks": n_seed_signal_clicks_1000,
                        "nr_of_test_set_song_recommendations_by_scenario": n_seed_ground_truth_size_1000,
                        "test_set_constraint": test_set_constraint,
                        "nr_of_considered_test_set_playlists": sum(test_set_playlists_to_consider),
                        "nr_of_targeted_seed_songs": nr_of_targeted_seed_songs,
                        "seed_song_recs_in_holdouts_1000": n_seed_seed_song_recs_in_holdouts_1000
                    }
                    results.append(combined_scenario_results)

    
        # if it's not a baseline, we plot the performance
        sns.set()
        sns.set_palette("bright")
        cp = sns.color_palette()
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        sns.lineplot(ax=axes[0, 0], x=data_manager.N_SEED_SONGS, y=precs, label=args.model_name, markers=True, linewidth=2.0, color=cp[0])
        sns.lineplot(ax=axes[0, 1], x=data_manager.N_SEED_SONGS, y=recalls, label=args.model_name, markers=True, linewidth=2.0, color=cp[0])
        sns.lineplot(ax=axes[0, 2], x=data_manager.N_SEED_SONGS, y=R_precs, label=args.model_name, markers=True, linewidth=2.0, color=cp[0])
        sns.lineplot(ax=axes[1, 0], x=data_manager.N_SEED_SONGS, y=ndcgs, label=args.model_name, markers=True, linewidth=2.0, color=cp[0])
        sns.lineplot(ax=axes[1, 1], x=data_manager.N_SEED_SONGS, y=clicks, label=args.model_name, markers=True, linewidth=2.0, color=cp[0])
        sns.lineplot(ax=axes[1, 2], x=data_manager.N_SEED_SONGS, y=norm_pop, label=args.model_name, markers=True, linewidth=2.0, color=cp[0])

        handles, labels = axes[1, 2].get_legend_handles_labels()
        fig.legend(handles, labels, loc='right', borderaxespad=0.3)
        for row in tqdm.tqdm(axes, disable=disable_progress_bar, file=sys.stdout):
            for ax in row:
                ax.get_legend().remove()
                ax.set_xticks(data_manager.N_SEED_SONGS)
                ax.tick_params(axis="both", labelsize=14)
                ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

        axes[0, 0].set_ylabel("Precision (in %)", {"size": 14, 'weight': 'bold'})
        axes[0, 1].set_ylabel("Recall (in %)", {"size": 14, 'weight': 'bold'})
        axes[0, 2].set_ylabel("R-Precision (in %)", {"size": 14, 'weight': 'bold'})
        axes[1, 0].set_ylabel("NDCG (in %)", {"size": 14, 'weight': 'bold'})
        axes[1, 1].set_ylabel("Clicks (in number)", {"size": 14, 'weight': 'bold'})
        axes[1, 2].set_ylabel("Popularity (in %)", {"size": 14, 'weight': 'bold'})
        fig.text(0.5, 0.04, 'Number of seed songs', ha='center', size=18, weight='bold')
        plt.subplots_adjust(left=0.1,
                            bottom=0.1,
                            right=0.9,
                            top=0.9,
                            wspace=0.25,
                            hspace=0.15)

        figure_save_path = str(Path(outfile_path, f"all_results_epoch{epoch}.pdf"))
        print(f'Saving figure to {figure_save_path}')
        plt.savefig(figure_save_path, bbox_inches="tight")
    
    else:
        # calculate worst case results as baseline


        for test_set_constraint, targeted_base_songs in test_set_constraints.items():

            print("test_set_constraint:", test_set_constraint)
            print("targeted_base_songs:", targeted_base_songs)
            
            if test_set_constraint == "entire_test_set" and targeted_base_song_as_track_ids and len(targeted_base_song_as_track_ids) > 0:
                seed_song_recs_in_holdouts = test_evaluator.compute_all_seed_song_recs_in_holdouts(rec, targeted_base_song_as_track_ids)
                seed_song_recs_in_holdouts_1000 = seed_song_recs_in_holdouts.reshape(-1, 1000).sum(axis=1)
            else:
                seed_song_recs_in_holdouts = None
                seed_song_recs_in_holdouts_1000 = None
            
            # get the number of playlists (from the 10K total test set playlists) that are considered for the test set constraint
            test_set_playlists_to_consider = test_evaluator.get_test_set_playlists_to_consider(test_dataset, rec, test_set_constraint, targeted_base_songs)

            # Calculate average values for overall results
            results.append({
                "target_song": "-",
                "epoch": epoch,
                "is_baseline": baseline_name,
                "scenario": "all",
                "avg_r_precision": np.mean(R_precs),
                "avg_ndcg": np.mean(ndcgs),
                "avg_recommended_song_clicks": np.mean(clicks),
                "total_signal_recommendations": 0,
                "total_signal_recommendations_in_holdouts": 0,
                "avg_rank_signal": 0,
                "avg_rank_signal_in_holdouts": 0,
                "avg_signal_song_clicks": 0,
                "nr_of_test_set_song_recommendations_by_scenario": sum(ground_truth_size_1000),
                "test_set_constraint": test_set_constraint,
                "nr_of_considered_test_set_playlists": sum(test_set_playlists_to_consider),
                "nr_of_targeted_seed_songs": nr_of_targeted_seed_songs,
                "seed_song_recs_in_holdouts_1000": sum(seed_song_recs_in_holdouts_1000) if seed_song_recs_in_holdouts_1000 is not None else None
                #"avg_precision": np.mean(precs),
                #"avg_recall": np.mean(recalls),
                #"avg_recall": np.mean(recalls),
                #"popularity": np.mean(norm_pop),
            })

    return results


@utils.timer
def get_baseline_recos(rec, rec_without_collective_action, gt, gt_without_collective_action, signal_as_track_ids):
    baseline_recos = {}
    bl_replace_first_relevant = []
    bl_replace_first_relevant_if_target_rec = []
    bl_replace_first_relevant_if_target_rec_in_holdout = []
    bl_replace_first_relevant_if_target_rec_in_holdout_random = []


    for rec_ca, rec_wo_ca, gt_with_ca, gt_wo_ca in zip(rec, rec_without_collective_action, gt, gt_without_collective_action):
        # Copy the original recommendations without collective action
        modified_playlist = copy.deepcopy(rec_wo_ca)
        modified_playlist_if_target_rec = copy.deepcopy(rec_wo_ca)
        modified_playlist_if_target_rec_in_holdout = copy.deepcopy(rec_wo_ca)
        modified_playlist_if_target_rec_in_holdout_random = copy.deepcopy(rec_wo_ca)
        
        for signal_as_track_id in signal_as_track_ids:
            # Replace the first relevant song with the signal
            for i, track in enumerate(rec_wo_ca):
                if track in gt_wo_ca: # check if rec without collective action is a relevant song
                    modified_playlist[i] = -1  # Replacement first relevant song
                    if signal_as_track_id in rec_ca: # check if signal is in rec with collective action
                        modified_playlist_if_target_rec[i] = -1  # Replace relevant song if collective was indeed successful for this playlist
                        if np.where(rec_ca == signal_as_track_id)[0][0] < len(gt_with_ca): # len(gt_with_ca) == len(gt_wo_ca)
                            modified_playlist_if_target_rec_in_holdout[i] = -1  # Replacement if signal is in rec within the holdout songs
                    break
            # now the random replacement
            if signal_as_track_id in rec_ca: # check if signal is in rec with collective action
                if np.where(rec_ca == signal_as_track_id)[0][0] < len(gt_with_ca): # len(gt_with_ca) == len(gt_wo_ca)
                    # get random index of song in modified_playlist_if_target_rec_in_holdout_random
                    # get random int base on len of modified_playlist_if_target_rec_in_holdout_random
                    random_index = np.random.randint(0, len(modified_playlist_if_target_rec_in_holdout_random))
                    modified_playlist_if_target_rec_in_holdout_random[random_index] = -1  # Replacement if signal is in rec within the holdout songs


        bl_replace_first_relevant.append(modified_playlist)
        bl_replace_first_relevant_if_target_rec.append(modified_playlist_if_target_rec)
        bl_replace_first_relevant_if_target_rec_in_holdout.append(modified_playlist_if_target_rec_in_holdout)
        bl_replace_first_relevant_if_target_rec_in_holdout_random.append(modified_playlist_if_target_rec_in_holdout_random)

    baseline_recos["bl"] = rec_without_collective_action
    baseline_recos["bl_replace_first_relevant"] = np.array(bl_replace_first_relevant)
    #baseline_recos["bl_replace_first_relevant_if_target_rec"] = np.array(bl_replace_first_relevant_if_target_rec)
    baseline_recos["bl_replace_first_relevant_if_target_rec_in_holdout"] = np.array(bl_replace_first_relevant_if_target_rec_in_holdout)
    baseline_recos["bl_replace_first_relevant_if_target_rec_in_holdout_random"] = np.array(bl_replace_first_relevant_if_target_rec_in_holdout_random)

    return baseline_recos


def get_performative_baseline_recos(rec, gt, signal_as_track_ids, performativity):
    rng = np.random.default_rng(seed=42)
    for rec_ca, gt_with_ca in zip(rec, gt):
        for signal_as_track_id in signal_as_track_ids:
            # Replace the signal with a relevant song with probability performativity, using rng
            for i, track in enumerate(rec_ca):
                if track == signal_as_track_id: # check if rec with collective action is the signal
                    if rng.random() <= performativity:
                        # get relevant song from the ground truth that is not in the rec with collective action yet
                        for track in gt_with_ca:
                            if track not in rec_ca:
                                rec_ca[i] = track
                                break
    return rec

def get_single_holdout_recos(recos, gt):
    R = len(gt)
    if R == 0:
        return []
    else:
        return recos[:R]


@utils.timer
def plot_winners_and_loosers(rec, rec_without_collective_action, gt, data_manager, data_manager_without_collective_action):
    
    # first, for each prediction, only get the number of masked songs per playlist, and not the full 500 predictions
    n = len(gt)
    rec = np.array([get_single_holdout_recos(rec[i], gt[i]) for i in range(n)])
    rec_without_collective_action = np.array([get_single_holdout_recos(rec_without_collective_action[i], gt[i]) for i in range(n)])

    tracks_uri_by_id = {value['id']: value['track_uri'] for key, value in data_manager.tracks_info.items()}
    tracks_uri_by_id_without_collective_action = {value['id']: value['track_uri'] for key, value in data_manager_without_collective_action.tracks_info.items()}
    artist_name_by_id = {value: key for key, value in data_manager.artist_ids.items()}
    artist_name_by_id_without_collective_action = {value: key for key, value in data_manager_without_collective_action.artist_ids.items()}

    # get artist for each recommendation
    artist_rec = []
    artist_rec_without_collective_action = []
    artist_name_by_uri = {}
    track_rec = []
    track_rec_without_collective_action = []
    track_name_by_uri = {}
    artist_uri_by_track_uri = {}
    
    for r in rec:
        for track_id in r:
            track_uri = tracks_uri_by_id[track_id].replace("spotify:track:", "")
            artist_name = data_manager.tracks_info[track_uri]['artist_name']
            artist_uri = data_manager.tracks_info[track_uri]['artist_uri']
            artist_name_by_uri[artist_uri] = artist_name
            artist_rec.append(artist_uri)
            
            track_name = data_manager.tracks_info[track_uri]['track_name']
            track_name_by_uri[track_uri] = track_name
            artist_uri_by_track_uri[track_uri] = artist_uri
            track_rec.append(track_uri)
    
    for r in rec_without_collective_action:
        for track_id in r:
            track_uri = tracks_uri_by_id_without_collective_action[track_id].replace("spotify:track:", "")
            artist_name = data_manager_without_collective_action.tracks_info[track_uri]['artist_name']
            artist_uri = data_manager_without_collective_action.tracks_info[track_uri]['artist_uri']
            artist_name_by_uri[artist_uri] = artist_name
            artist_rec_without_collective_action.append(artist_uri)
    
            track_name = data_manager_without_collective_action.tracks_info[track_uri]['track_name']
            track_name_by_uri[track_uri] = track_name
            artist_uri_by_track_uri[track_uri] = artist_uri
            track_rec_without_collective_action.append(track_uri)
    
    # count difference before and after collective action
    rec_counts = Counter(artist_rec)
    rec_without_collective_action_counts = Counter(artist_rec_without_collective_action)

    # Calculate rec_gain and rec_gain_relative for each artist
    winners_and_loosers = []
    for artist_uri in set(artist_rec + artist_rec_without_collective_action):
        artist_rec_without_collective_action = rec_without_collective_action_counts[artist_uri]
        artist_rec_with_collective_action = rec_counts[artist_uri]
        rec_gain = artist_rec_with_collective_action - artist_rec_without_collective_action
        rec_gain_relative = (artist_rec_with_collective_action / artist_rec_without_collective_action - 1) if artist_rec_without_collective_action > 0 else 0
        artist_name = artist_name_by_uri[artist_uri]
        winners_and_loosers.append({
            "artist_uri": artist_uri,
            "artist_name": artist_name.replace("$", "\\$"),
            "rec_without_collective_action": artist_rec_without_collective_action,
            "rec_with_collective_action": artist_rec_with_collective_action,
            "rec_gain": rec_gain,
            "rec_gain_relative": rec_gain_relative
        })
    
    # do the same for tracks
    track_rec_counts = Counter(track_rec)
    track_rec_without_collective_action_counts = Counter(track_rec_without_collective_action)

    # Calculate rec_gain and rec_gain_relative for each artist
    winners_and_loosers_track_level = []
    for track_uri in set(track_rec + track_rec_without_collective_action):
        artist_uri = artist_uri_by_track_uri[track_uri]
        #artist_rec_without_collective_action = rec_without_collective_action_counts[artist_uri]
        #artist_rec_with_collective_action = rec_counts[artist_uri]
        track_rec_without_collective_action = track_rec_without_collective_action_counts[track_uri]
        track_rec_with_collective_action = track_rec_counts[track_uri]
        rec_gain = track_rec_with_collective_action - track_rec_without_collective_action
        rec_gain_relative = (track_rec_with_collective_action / track_rec_without_collective_action - 1) if track_rec_without_collective_action > 0 else 0
        artist_name = artist_name_by_uri[artist_uri]
        track_name = track_name_by_uri[track_uri]
        winners_and_loosers_track_level.append({
            "track_uri": track_uri,
            "track_name": track_name.replace("$", "\\$"),
            "artist_uri": artist_uri,
            "artist_name": artist_name.replace("$", "\\$"),
            "rec_without_collective_action": track_rec_without_collective_action,
            "rec_with_collective_action": track_rec_with_collective_action,
            "rec_gain": rec_gain,
            "rec_gain_relative": rec_gain_relative
        })

    return winners_and_loosers, winners_and_loosers_track_level




    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--metric", type = str, required = True,
                    help = "metric to compute")
    parser.add_argument('-strategy', '--signal_planting_strategy', type=str, required=True)
    parser.add_argument('-b', '--budget', type=int, help='the number of signals to plant.')
    parser.add_argument('-s', '--seed', type=int, required=False, help='seed for random number generation.')
    parser.add_argument('-collr', '--collective_requirement', type=str, required=False, default="", help='empty string if collective is random subset or track uri if every collective playlist must contain this track.')

    parser.add_argument('--algo', type=str, required=True)
    parser.add_argument('--submission_file', type=str, required=True)
    parser.add_argument('--outfile', type=str, required=False)
    parser.add_argument('--dataset_size', type=str, required=True)
    parser.add_argument('--fold', type=str, required=True)
    parser.add_argument('--challenge_dataset_name', type=str, required=True)
    parser.add_argument('--base_path', type=str, required=False)
    parser.add_argument('--where_to_save_results', type=str, required=False, default='submission_results_deezer.csv', help='Path to save the results')
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument("--params_file", type = str, required = False, help = "File for hyperparameters", default = "resources/params/best_params_rta.json")

    # unused arguments
    parser.add_argument('--base_path_data', type=str, required=False)
    parser.add_argument('--dataset_name', type=str, required=False)

    save_winners_and_loosers_to_csv = True # TODO: make this an argument

    args = parser.parse_args()

    if "none" in args.signal_planting_strategy:
        manipulated_dataset_directory = args.signal_planting_strategy
    else:
        manipulated_dataset_directory = f"signal_planting_strategy_{args.signal_planting_strategy}_budget_{args.budget}{args.collective_requirement}"


    outfile_path = Path(args.base_path, "submissions_and_results", f"2023_deezer_{args.model_name}", args.dataset_size, manipulated_dataset_directory, args.fold)

    data_manager_path = str(Path(args.dataset_name, args.dataset_size, manipulated_dataset_directory, args.fold))

    signal_track_ids, track_id_by_track_uri, additionaltargetsong = get_target_song_info(args.signal_planting_strategy)
    signals = list(track_id_by_track_uri.keys())

    targeted_base_songs = None


    if "none" in manipulated_dataset_directory:
        # load the fold config
        fold_config_file_path = Path(args.base_path_data, "recommender_system/2023_deezer_transformers/resources/data/config_files", f"config_strategy_none_{args.fold}.yaml")
        #print("fold_config_file_path:", fold_config_file_path)
        with open(fold_config_file_path, "r") as fold_config_file:
            fc = munchify(yaml.safe_load(fold_config_file))
        fold_config = {
            "budget": "",
            "collective_action_description": "",
            "config_filename": "",
            "dataset_name": "",
            "fold": args.fold,
            "seed": 42,
            "signal": signals,
            "signal_planting_strategy": args.signal_planting_strategy,
            "collective_requirement": args.collective_requirement,
            "nr_of_train_set_songs": fc.nr_of_train_set_songs,
            "nr_of_test_set_song_recommendations": fc.nr_of_test_set_song_recommendations,
            #"targeted_base_songs": targeted_base_songs # can get very large
            "targeted_base_songs": None
        }
    else:
        # load the fold config
        fold_config_file_path = Path(args.base_path_data, "recommender_system/2023_deezer_transformers/resources/data/config_files", f"config_strategy_{args.signal_planting_strategy}_{args.budget}_{args.fold}{args.collective_requirement}.yaml")
        #print("fold_config_file_path:", fold_config_file_path)
        with open(fold_config_file_path, "r") as fold_config_file:
            fc = munchify(yaml.safe_load(fold_config_file))

        if args.budget != fc.budget:
            print(f"\n\n\n   >>>>> WARNING: budget argument ({args.budget}) does not match budget in fold config ({fc.budget}) ! ! ! <<<<<\n")
        

        onefamoussong = get_onefamoussong(args.signal_planting_strategy)
        severalfamoussongs = get_severalfamoussongs(args.signal_planting_strategy)
        if "onefamoussong" in args.signal_planting_strategy:
            targeted_base_songs = [onefamoussong]
        elif "severalfamoussongs" in args.signal_planting_strategy:
            targeted_base_songs = severalfamoussongs
        elif "optimal_bigram" in args.signal_planting_strategy:
            #load file from 
            targeted_base_songs_file = str(Path(outfile_path, "collective_strategy_bigram_seed_songs_and_top_k.csv"))
            # load as df
            targeted_base_songs = pd.read_csv(targeted_base_songs_file)["seed_song_uri"].unique().tolist()

            # TODO: load all targeted seed songs based on outfile_path?
        else:
            try:
                targeted_base_songs_file = str(Path(outfile_path, "targeted_seed_songs.csv"))
                targeted_base_songs = pd.read_csv(targeted_base_songs_file)["seed_song_uri"].unique().tolist()
            except:
                print(f"WARNING: targeted_base_songs.csv not found in {outfile_path}. continue with 'targeted_base_songs=None'\n")
        
        if targeted_base_songs:
            print(f"A total of {len(targeted_base_songs)} targeted base songs were found in {targeted_base_songs_file}.")
            print(f"unique targeted base songs: {len(set(targeted_base_songs))}")

        fold_config = {
            "budget": fc.budget,
            "collective_action_description": fc.collective_action_description,
            "config_filename": str(fold_config_file_path),
            "dataset_name": fc.dataset_name,
            "fold": f"{fc.fold}",
            "seed": fc.seed,
            "signal": signals,
            "signal_planting_strategy": fc.signal_planting_strategy,
            "collective_requirement": args.collective_requirement,
            "nr_of_train_set_songs": fc.nr_of_train_set_songs,
            "nr_of_test_set_song_recommendations": fc.nr_of_test_set_song_recommendations,
            #"targeted_base_songs": targeted_base_songs # can get very large
            "targeted_base_songs": None
        }


    foldername = str(Path(args.base_path_data, "recommender_system/2023_deezer_transformers/resources/data/"))
    data_manager = DataManager(replication_folder_name = data_manager_path, foldername = foldername, signal_planting_strategy=manipulated_dataset_directory)


    #### #### #### #### #### ####
    #### GET BASELINE RECOMMENDATIONS
    #### #### #### #### #### ####


    if "autoregressive" in args.signal_planting_strategy:
        strategy_no_col_action = "none_autoregressive"
    else:
        strategy_no_col_action = "none"
    

    data_manager_without_collective_action = DataManager(replication_folder_name = str(Path(args.dataset_name, "data", strategy_no_col_action, args.fold)), foldername = foldername, signal_planting_strategy=strategy_no_col_action)
    test_evaluator_without_collective_action, _ = data_manager_without_collective_action.get_test_data("test")

    # load recos without collective action
    outfile_no_collective_action = str(Path(args.base_path, "submissions_and_results", f"2023_deezer_{args.model_name}", "data", strategy_no_col_action, args.fold, args.outfile))
    rec_without_collective_action = np.load(("%s.npy") % (outfile_no_collective_action))

    #### #### #### #### #### ####

    # get track ids for all track uris
    if "none" in manipulated_dataset_directory:
        signal_as_track_ids = None
    else:
        tracks_id_by_uri = {value['track_uri']: value['id'] for key, value in data_manager.tracks_info.items()}
        # get track ids for all track uris
        signal_as_track_ids = []
        signal_by_track_id = {}
        for signal in signals:
            signal_as_track_id = tracks_id_by_uri[signal]
            signal_as_track_ids.append(signal_as_track_id)
            signal_by_track_id[signal_as_track_id] = signal
            
        
        targeted_base_song_as_track_ids = []
        if targeted_base_songs:
            for targeted_base_song in targeted_base_songs:
                try:
                    targeted_base_song_as_track_id = tracks_id_by_uri[targeted_base_song]
                    targeted_base_song_as_track_ids.append(targeted_base_song_as_track_id)
                    #print("check targeted_base_song:", targeted_base_song)
                    #print("check more:", "targeted_base_song_as_track_id:", targeted_base_song_as_track_id)
                    #print("    >>>>>targeted_base_song_as_track_ids:", targeted_base_song_as_track_ids)
                except:
                    print(f"WARNING: targeted_base_song {targeted_base_song} not found in tracks_id_by_uri. continue without this song.\n")


        signal_as_track_ids_without_collective_action = None
        tracks_id_by_uri_without_collective_action = {value['track_uri']: value['id'] for key, value in data_manager_without_collective_action.tracks_info.items()}
        # get the track ids for the targeted base songs but as in the baseline rec_without_collective_action
        targeted_base_song_as_track_ids_without_collective_action = []
        if targeted_base_songs:
            for targeted_base_song in targeted_base_songs:
                try:
                    targeted_base_song_as_track_id_without_collective_action = tracks_id_by_uri_without_collective_action[targeted_base_song]
                    targeted_base_song_as_track_ids_without_collective_action.append(targeted_base_song_as_track_id_without_collective_action)
                except:
                    print(f"WARNING: targeted_base_song {targeted_base_song} not found in tracks_id_by_uri. continue without this song.\n")

    assert "none" in manipulated_dataset_directory or signal_as_track_ids is not None, f"ERROR: signal track id not found for manipulated_dataset_directory={manipulated_dataset_directory}"

    test_evaluator, test_dataloader = data_manager.get_test_data("test")


    # List to store the loaded arrays and their corresponding epochs
    recs_of_all_epochs = []
    for filename in os.listdir(outfile_path):
        if filename.startswith(args.outfile) and filename.endswith(".npy"):
            epoch = filename.replace(args.outfile, "").replace(".npy", "").replace("_epoch", "")
            if epoch == "":
                epoch = "all"
            else:
                epoch = int(epoch) + 1
            rec = np.load(str(Path(outfile_path, filename)))
            recs_of_all_epochs.append((epoch, rec))

    # Custom sorting function
    def custom_sort(item):
        epoch, _ = item
        if epoch == "all":
            return float('-inf')  # Treat "all" as the smallest value for sorting purposes
        return -epoch  # Reverse sorting for numeric values

    # Sort the arrays by epoch with the custom function in descending order
    recs_of_all_epochs.sort(key=custom_sort)

    results = []

    # Loop through the loaded recs and calculate the results
    for epoch, rec in recs_of_all_epochs:
        print(f"\n\n >>>> Processing recommendations from epoch {epoch} <<<<\n\n")

        #### #### #### #### #### ####
        #### GET BASELINE RECOMMENDATIONS
        #### #### #### #### #### ####

        # compare recos
        # get some baseline recos, i.e., worst case performance for collective action relative to the success

        if "none" not in manipulated_dataset_directory:
            baseline_recos = get_baseline_recos(rec, rec_without_collective_action, test_evaluator.gt, test_evaluator_without_collective_action.gt, signal_as_track_ids)
            baseline_recs = (baseline_recos["bl"], targeted_base_song_as_track_ids_without_collective_action, EvaluationDataset(data_manager_without_collective_action, data_manager_without_collective_action.test_indices))
        else:
            baseline_recs = None
            signal_by_track_id = None
            targeted_base_song_as_track_ids = None

        #### #### #### #### #### ####


        results.extend(calculate_results(rec, epoch, test_evaluator, data_manager, outfile_path, args, signal_as_track_ids=signal_as_track_ids, signal_by_track_id=signal_by_track_id, targeted_base_song_as_track_ids=targeted_base_song_as_track_ids, baseline_recs = baseline_recs))

        if "none" not in manipulated_dataset_directory:
            
            #### #### #### #### #### ####
            #### FIRST RUN THE PERFORMATIVE BASELINE
            #### #### #### #### #### ####

            performativity=1
            print(f" Calculating performative baseline with performativity={performativity}")
            rec_performative_baseline = get_performative_baseline_recos(copy.deepcopy(rec), test_evaluator.gt, signal_as_track_ids, performativity)

            print(" calculating results for performative_baseline recos")
            results.extend(calculate_results(rec_performative_baseline, epoch, test_evaluator, data_manager, outfile_path, args, baseline_name="performative_baseline", signal_as_track_ids=signal_as_track_ids, signal_by_track_id=signal_by_track_id))
            print(" done calculating results for performative_baseline recos")

            #### #### #### #### #### ####
            
            
            # load testset without collective action
            
            solution_is_comparable_with_none = False
            # compare tests sets
            
            if test_evaluator.test_size == test_evaluator_without_collective_action.test_size and all(len(test_evaluator.gt[i]) == len(test_evaluator_without_collective_action.gt[i]) for i in range(len(test_evaluator.gt))) and test_evaluator.n_recos == test_evaluator_without_collective_action.n_recos:
                solution_is_comparable_with_none = True

                print("calculating results for baseline recos")
                for baseline_name, baseline_rec in baseline_recos.items():
                    results.extend(calculate_results(baseline_rec, epoch, test_evaluator_without_collective_action, data_manager_without_collective_action, outfile_path, args, baseline_name=baseline_name, targeted_base_song_as_track_ids=targeted_base_song_as_track_ids_without_collective_action))
                print("done calculating results for baseline recos")
                
            else:
                # CANNOT COMPUTE PERFORMANCE LOSS SINCE TEST SET IS NOT COMPARABLE with none
                print("WARNING: test set is not comparable with none. here's why:")
                print("    check if test_size equal:", test_evaluator.test_size == test_evaluator_without_collective_action.test_size)
                print("    check if gt equal:", all(len(test_evaluator.gt[i]) == len(test_evaluator_without_collective_action.gt[i]) for i in range(len(test_evaluator.gt))))
                print("    check if n_recos equal:", test_evaluator.n_recos == test_evaluator_without_collective_action.n_recos)
            

        # Write results to CSV
        csv_file_path = Path(args.base_path, "submissions_and_results", f"{args.where_to_save_results}.csv")
        headers = sorted(list(results[0].keys()) + list(fold_config.keys())) + ["algo", "submission_results_path", "challenge_dataset_path", "submission_file", "dataset_size", "manipulated_dataset_directory"]
        if not os.path.isfile(csv_file_path):
            print(f"Writing headers to {csv_file_path}")
            with open(csv_file_path, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(headers)

        print(f"Writing results to {csv_file_path}")
        # Append the new line with results to the csv file
        with open(csv_file_path, 'a', newline='') as file:
            writer = csv.writer(file)
            for result in results:
                row = [value for key, value in sorted({**result, **fold_config}.items())] + [args.algo, "", "", args.submission_file, args.dataset_size, manipulated_dataset_directory]
                writer.writerow(row)



        if "none" not in manipulated_dataset_directory and save_winners_and_loosers_to_csv and solution_is_comparable_with_none and epoch == "all":
            #### #### #### #### #### ####
            #### PLOT ADVERSE EFFECTS ###
            #### #### #### #### #### ####
            

            print("ok1")
            winners_and_loosers, winners_and_loosers_track_level = plot_winners_and_loosers(rec, rec_without_collective_action, test_evaluator.gt, data_manager, data_manager_without_collective_action)

            # Write artist level results to CSV
            csv_file_path = Path("/fast/jbaumann/spotify-algo-collective-action/recommender_system/2023_deezer_transformers/resources/data/winners_and_losers", f"winners_and_loosers_{args.where_to_save_results}_{manipulated_dataset_directory}_{args.fold}")
            headers = ["artist_uri", "artist_name", "rec_without_collective_action", "rec_with_collective_action", "rec_gain", "rec_gain_relative"] + sorted(list(fold_config.keys())) + ["algo", "submission_file", "dataset_size", "manipulated_dataset_directory"]
            if not os.path.isfile(csv_file_path):
                print(f"Writing headers to {csv_file_path}")
                with open(csv_file_path, 'w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(headers)

            print(f"Writing winners_and_losers results to {csv_file_path}")
            # Append the new line with results to the csv file
            with open(csv_file_path, 'a', newline='') as file:
                writer = csv.writer(file)
                for wl in winners_and_loosers:
                    row = [wl["artist_uri"], wl["artist_name"], wl["rec_without_collective_action"], wl["rec_with_collective_action"], wl["rec_gain"], wl["rec_gain_relative"]] + [value for key, value in sorted(fold_config.items())] + [args.algo, args.submission_file, args.dataset_size, manipulated_dataset_directory]
                    writer.writerow(row)

            # Write song level results to CSV
            csv_file_path = Path("/fast/jbaumann/spotify-algo-collective-action/recommender_system/2023_deezer_transformers/resources/data/winners_and_losers", f"winners_and_loosers_track_level_{args.where_to_save_results}_{manipulated_dataset_directory}_{args.fold}")
            headers = ["track_uri", "track_name", "artist_uri", "artist_name", "rec_without_collective_action", "rec_with_collective_action", "rec_gain", "rec_gain_relative"] + sorted(list(fold_config.keys())) + ["algo", "submission_file", "dataset_size", "manipulated_dataset_directory"]
            if not os.path.isfile(csv_file_path):
                print(f"Writing headers to {csv_file_path}")
                with open(csv_file_path, 'w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(headers)

            print(f"Writing winners_and_losers SONG LEVEL results to {csv_file_path}")
            # Append the new line with results to the csv file
            with open(csv_file_path, 'a', newline='') as file:
                writer = csv.writer(file)
                for wl in winners_and_loosers_track_level:
                    row = [wl["track_uri"], wl["track_name"], wl["artist_uri"], wl["artist_name"], wl["rec_without_collective_action"], wl["rec_with_collective_action"], wl["rec_gain"], wl["rec_gain_relative"]] + [value for key, value in sorted(fold_config.items())] + [args.algo, args.submission_file, args.dataset_size, manipulated_dataset_directory]
                    writer.writerow(row)

    print("done :)")