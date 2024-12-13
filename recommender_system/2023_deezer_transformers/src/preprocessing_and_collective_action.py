
import sys
import json
import re
import os
import datetime
import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse import save_npz
import csv
import tqdm
import pickle
import argparse
from src.data_manager.data_manager import DataManager
from collections import defaultdict, Counter
from pathlib import Path
import yaml
from utils import get_nr_of_unique_tracks, get_target_song_info, get_promotedifferentsongs, get_train_fraction, get_overshoot_int, get_plantseed, get_insertbefore, get_replaceseed, get_duplicateseed, get_onefamoussong, get_severalfamoussongs, get_song_to_promote, disable_progress_bar
import time
import psutil
from collective_action import plant_signal, save_data_with_signal
import math
import pdb
import copy

total_playlists = 0
total_tracks = 0
unique_track_count = 0
tracks = set()
artists = set()
albums = set()
titles = set()
total_descriptions = 0
ntitles = set()
n_playlists = 1000000
#n_tracks = 2262292
#playlist_track = lil_matrix((n_playlists, n_tracks), dtype=np.int32) # to build interaction matrix of binary value
tracks_info = {} # to keep base infos on tracks
title_histogram = Counter()
artist_histogram = Counter()
track_histogram = Counter()
last_modified_histogram = Counter()
num_edits_histogram = Counter()
playlist_length_histogram = Counter()
num_followers_histogram = Counter()
playlists_list = []
quick = False
max_files_for_quick_processing = 50



def process_mpd(raw_path, out_path, args, replication_folder_name, foldername):


    ######## ######## ######## ########
    ######## COLLECTIVE ACTION ########
    ######## ######## ######## ########

    signal_metadata = {
        "artist_name": "AlgoCollective",
        "artist_uri": "spotify:artist:AlgoCollectiveURI",
        "track_name": "AlgoCollectiveSong",
        "album_uri": "spotify:album:NewAlbumByAlgoCollectiveURI",
        "duration_ms": 200000,
        "album_name": "NewAlbumByAlgoCollective"
    }


    if "none" in args.signal_planting_strategy:
        # budget = 0
        budget = args.budget # TODO: just a test for empty playlist, remove again later
        track_id_by_track_uri = {}
        track_id = 0
        known_fraction_of_train_set = 1
        onefamoussong = None
        severalfamoussongs = None
        promoteexisting = None
    else:
        budget = args.budget
        known_fraction_of_train_set = get_train_fraction(args.signal_planting_strategy)
        print("known_fraction_of_train_set:", known_fraction_of_train_set)
        signal_track_ids, track_id_by_track_uri, additionaltargetsong = get_target_song_info(args.signal_planting_strategy)
        overshoot = get_overshoot_int(args.signal_planting_strategy)
        plantseed = get_plantseed(args.signal_planting_strategy)
        replaceseed = get_replaceseed(args.signal_planting_strategy)
        insertbefore = get_insertbefore(args.signal_planting_strategy)
        duplicateseed = get_duplicateseed(args.signal_planting_strategy)
        onefamoussong = get_onefamoussong(args.signal_planting_strategy)
        severalfamoussongs = get_severalfamoussongs(args.signal_planting_strategy)
        promoteexisting = get_song_to_promote(args.signal_planting_strategy)
        signal = list(track_id_by_track_uri.keys())[0]
        print("check signal:", signal)
        print("check signals (uris and ids):", track_id_by_track_uri)
        print("signal_track_ids:", signal_track_ids)
        print("additionaltargetsong:", additionaltargetsong)
        print(f" heyhey onefamoussong: {onefamoussong}")
        print(f" heyhey severalfamoussongs: {severalfamoussongs}")
        #pdb.set_trace()
        track_id = len(signal_track_ids)


    ######## ######## ######## ########
    ######## ----------------- ########
    ######## ######## ######## ########

    tracks_with_fake_metadata = set()


    print("processing MPD to organize collective and to define train, eval, and test set indices")
    global playlists_list
    count = 0
    all_playlists = []
    all_playlists_only_track_ids = []
    all_song_info = {}

    
    filenames = os.listdir(raw_path)
    for filename in tqdm.tqdm(sorted(filenames, key=str), disable=disable_progress_bar, file=sys.stdout):
        #print("filename:", filename)
        if filename.startswith("mpd.slice.") and filename.endswith(".json"):
            #playlists_list = []
            fullpath = os.sep.join((raw_path, filename))
            f = open(fullpath)
            js = f.read()
            f.close()
            mpd_slice = json.loads(js)
            #process_info(mpd_slice["info"])

            playlists_one_file = mpd_slice["playlists"]
            
            # TODO: the following for loop generates fake metadata for some songs in the dataset
            # TODO: remove again?
            #for pl in playlists_one_file:
            #    # loop through songs in playlist and homogenize the metadata
            #    for track in pl["tracks"]:
            #        # with some probability p, set track metadata to signal_metadata
            #        #if np.random.uniform() < 0.1 or track["track_uri"] in tracks_with_fake_metadata:
            #        #tracks_with_fake_metadata.add(track["track_uri"])
            #        track["artist_name"] = signal_metadata["artist_name"]
            #        track["artist_uri"] = signal_metadata["artist_uri"]
            #        track["track_name"] = signal_metadata["track_name"]
            #        track["album_uri"] = signal_metadata["album_uri"]
            #        track["duration_ms"] = signal_metadata["duration_ms"]
            #        track["album_name"] = signal_metadata["album_name"]


            for playlist in playlists_one_file:
                tracks = []
                for track in playlist['tracks']:
                    track_uri = track['track_uri']
                    # add info to all_song_info if not there yet
                    if track_uri not in track_id_by_track_uri:
                        track_id_by_track_uri[track_uri] = track_id
                        all_song_info[track_id] = track
                        tracks.append(track_id)
                        track_id += 1
                    else:
                        tracks.append(track_id_by_track_uri[track_uri])
                all_playlists_only_track_ids.append(tracks)
            
            #all_playlists_only_track_ids.extend([[track['track_uri'] for track in playlist['tracks']] for playlist in playlists_one_file])

            all_playlists.extend(playlists_one_file)
            #for playlist in mpd_slice["playlists"]:
            #    process_playlist(playlist)
            count += 1
            #seqfile = open('%s/playlists_seq.csv' % out_path, 'a', newline ='')
            #with seqfile:
            #  write = csv.writer(seqfile) 
            #  write.writerows(playlists_list)  
            if quick and count > max_files_for_quick_processing:
                break
            #print("number of playlists:", len(all_playlists))
            # flush print statements
            sys.stdout.flush()
        #print("done with file:", filename)
        sys.stdout.flush()
    print("done with all files")
    sys.stdout.flush()
    print("done with all files2", flush=True)



    ######## ######## ######## ########
    ######## COLLECTIVE ACTION ########
    ######## ######## ######## ########

    rng = np.random.default_rng(seed=args.seed + int(args.fold.split("_")[1]))


    parts = args.signal_planting_strategy.split("_")
    # Determine beta that defines how many samples of the validation set are controlled by the collective
    beta_parts = [part for part in parts if "beta" in part]
    if beta_parts:  # Check if beta is in the strategy
        beta_str = beta_parts[0] # Take the first match
        beta = int(beta_str.replace("beta", ""))
        total_budget_train_set = math.ceil(budget * (1 - beta/100))
        total_budget_val_set = budget - total_budget_train_set
        print(f"beta is: {beta}")
        print(f"total budget is: {budget} (budget: {budget} beta fraction {beta}/100 are from val set)")
    else:
        beta = None
        total_budget_train_set = budget
        print(f"total budget is: {budget} (budget: {budget}, randomly sampled from train and eval set)")

    candidate_playlistpids = sorted([playlist["pid"] for playlist in all_playlists if len( set([t["track_uri"] for t in playlist["tracks"]]) ) > 20]) # find all playlists that have at least 20 unique songs
    # select only playlists containing song 
    #candidate_playlistpids = sorted([playlist["pid"] for playlist in all_playlists if (len( set([t["track_uri"] for t in playlist["tracks"]]) ) > 20) and any( ["7yyRTcZmCiyzzJlNzGC9Ol" in t["track_uri"] for t in playlist["tracks"]]) ]  )
    
    # get test set indices
    test_indices = rng.choice(candidate_playlistpids, 10000, replace=False)

    # TODO: test how often the famous song occurs in the test set
    seed_songs_in_testset_seed = defaultdict(lambda: 0)
    if onefamoussong or severalfamoussongs:
        for playlist in all_playlists:
            if playlist["pid"] in test_indices:
                for track in playlist["tracks"]:
                    if onefamoussong and track["track_uri"] == onefamoussong:
                        seed_songs_in_testset_seed[onefamoussong] += 1
                    elif severalfamoussongs and track["track_uri"] in severalfamoussongs:
                        seed_songs_in_testset_seed[track["track_uri"]] += 1
    
        print("seed_songs_in_testset_seed:", seed_songs_in_testset_seed)
        print("\n\n\n\n\n\n\n\n\n\n")
        print("---")
        print("test_indices:")
        print(test_indices)
        print("---")
        print("\n\n\n\n\n\n\n\n\n\n")
        
    # remove test set indices from candidate_playlistpids
    candidate_playlistpids_without_test = list(set(candidate_playlistpids) - set(test_indices))

    # get val set indices
    val_indices = rng.choice(candidate_playlistpids_without_test, 10000, replace=False)
    
    if beta is not None:
        # if beta is defined, remove val set indices from candidate_playlistpids_without_test
        candidate_playlistpids_without_test = list(set(candidate_playlistpids) - set(test_indices) - set(val_indices))

    if args.collective_requirement != "":
        # only take playlists containing broccoli and with at least 20 songs
        collective_candidates = sorted([playlist["pid"] for playlist in all_playlists if (len( set([t["track_uri"] for t in playlist["tracks"]]) ) > 20) and any( ["7yyRTcZmCiyzzJlNzGC9Ol" in t["track_uri"] for t in playlist["tracks"]]) ]  )
    else:
        if len(candidate_playlistpids_without_test) < budget: # TODO: for beta, this should test number of candidates in train and val for train and val budgets
            print(f"WARNING: budget {budget} is larger than number of playlists with more than 20 songs ({len(candidate_playlistpids_without_test)}). Setting collective_candidates to any playlist that is not in test")
            if beta is None:
                collective_candidates = list(set(range(len(all_playlists))) - set(test_indices))
            else:
                collective_candidates = list(set(range(len(all_playlists))) - set(test_indices) - set(val_indices))
            print("new length of collective_candidates:", len(collective_candidates))
        else:
            collective_candidates = candidate_playlistpids_without_test
    collective_playlistpids = rng.choice(collective_candidates, total_budget_train_set, replace=False)

    if beta is not None:
        # chosse additional beta indices from the val set, which are controlled by collective
        collective_playlistpids = np.concatenate([collective_playlistpids, rng.choice(val_indices, total_budget_val_set, replace=False)])

    tmp_trainval_indices = set(range(len(all_playlists))) - set(test_indices)

    train_indices = list(tmp_trainval_indices - set(val_indices))

    train_indices_not_controlled_by_collective = list(tmp_trainval_indices - set(collective_playlistpids))

    if known_fraction_of_train_set == 1:
        known_training_data_playlist_pids = tmp_trainval_indices
    elif known_fraction_of_train_set > 0:
        known_training_data_playlist_pids = np.concatenate([collective_playlistpids, rng.choice(train_indices_not_controlled_by_collective, math.ceil(known_fraction_of_train_set * len(train_indices_not_controlled_by_collective)), replace=False)])
    else:
        known_training_data_playlist_pids = collective_playlistpids
    
    print(f"selected {len(known_training_data_playlist_pids)} in total (from {len(train_indices_not_controlled_by_collective)} train&val playlists not controlled by collective plus {budget} controlled by collective) to be known for strategy engineering (trainfraction: {known_fraction_of_train_set})")
    
    
    # get the subset of X forming the collective, by the playlistpids
    collective = []
    known_training_data = []
    unknown_training_data = []

    if "promoteexisting" in args.signal_planting_strategy:
        # promote an existing song, so the collective should only get those playlists that contain the promoted song
        total_playlists = len(all_playlists)
        update_interval = total_playlists // 100  # Update every 1%
        for i, playlist in enumerate(all_playlists):
            if playlist["pid"] in collective_playlistpids and any( [promoteexisting in t["track_uri"] for t in playlist["tracks"]]):
                collective.append(all_playlists_only_track_ids[i])
                known_training_data.append(all_playlists_only_track_ids[i])
            else:
                known_training_data.append(all_playlists_only_track_ids[i])

            # Print status update
            if (i + 1) % update_interval == 0 or i == total_playlists - 1:
                print(f"Processed {i + 1} out of {total_playlists} playlists ({(i + 1) / total_playlists * 100:.2f}%)")

    else:
        total_playlists = len(all_playlists)
        update_interval = total_playlists // 100  # Update every 1%
        for i, playlist in enumerate(all_playlists):
            if playlist["pid"] in collective_playlistpids:
                collective.append(all_playlists_only_track_ids[i])
                known_training_data.append(all_playlists_only_track_ids[i])
            elif playlist["pid"] in known_training_data_playlist_pids:
                known_training_data.append(all_playlists_only_track_ids[i])
            elif known_fraction_of_train_set < 1 and playlist["pid"] in train_indices_not_controlled_by_collective:
                unknown_training_data.append(all_playlists_only_track_ids[i])

            # Print status update
            if (i + 1) % update_interval == 0 or i == total_playlists - 1:
                print(f"Processed {i + 1} out of {total_playlists} playlists ({(i + 1) / total_playlists * 100:.2f}%)")

        
        assert len(collective) == len(collective_playlistpids), f"Expected {len(collective_playlistpids)} playlists in the collective, but got {len(collective)} playlists in the collective"
        assert len(known_training_data) == len(known_training_data_playlist_pids), f"Expected {len(known_training_data_playlist_pids)} playlists in the known training data, but got {len(known_training_data)} playlists"
        assert len(known_training_data) >= len(collective_playlistpids), f"Expected {len(known_training_data)} >= {len(collective_playlistpids)}"


    print(f"number of playlists: {len(all_playlists)}", flush=True)
    # TODO: undo after debugging
    #print("number of playlists with more than 20 tracks:", len(sorted([playlist["pid"] for playlist in all_playlists if len( set([t["track_uri"] for t in playlist["tracks"]]) ) > 20])))
    print("number of playlists with more than 20 tracks:", len(candidate_playlistpids), "excluding those in test set, this gives", len(collective_candidates))
    if args.collective_requirement != "":
        print("\n    !!!!number of playlists with more than 20 tracks AND CONTAINING BROCCOLI:", len(collective_candidates), "!!!\n")
    # get overlap between collective and val set
    playlists_in_val_and_collective = set(val_indices).intersection(set(collective_playlistpids))
    print("number of playlists in collective:", len(collective_playlistpids), "of which", len(playlists_in_val_and_collective), "are from val set")
    print("number of playlists in test set:", len(test_indices))
    print("number of playlists in val set:", len(val_indices))
    print("number of playlists in train set:", len(train_indices))
    
    print("  number of playlists in known_training_data:", len(known_training_data))
    print("  number of playlists in unknown_training_data:", len(unknown_training_data))
    print("  number of playlists in collective_playlistpids:", len(collective_playlistpids))
    print("  number of playlists in known_training_data_playlist_pids:", len(known_training_data_playlist_pids))
    print("  number of playlists in train_indices_not_controlled_by_collective:", len(train_indices_not_controlled_by_collective))
    
    test_indices_set = set(test_indices)
    val_indices_set = set(val_indices)
    nr_of_train_set_songs = sum([len(set([t["track_uri"] for t in playlist["tracks"]])) for playlist in all_playlists if ((playlist["pid"] not in test_indices_set) and (playlist["pid"] not in val_indices_set))])
    nr_of_test_set_song_recommendations = int(sum([len(set([t["track_uri"] for t in playlist["tracks"]])) for playlist in all_playlists if playlist["pid"] in test_indices_set]) - 10000*5.5) # removing the total number seed songs
    print("NUMBER OF SONGS IN TRAIN SET:", nr_of_train_set_songs)
    print("NUMBER OF SONGS IN TEST SET including seed tracks:", int(nr_of_test_set_song_recommendations + 10000*5.5))
    print(f"NUMBER OF SONGS IN TEST SET minus seed tracks: {nr_of_test_set_song_recommendations}", flush=True)
    
    #assert len(collective_playlistpids) + len(test_indices) + len(val_indices) + len(train_indices) == n_playlists
    
    #pdb.set_trace()
    Path(foldername, "config_files").mkdir(parents=True, exist_ok=True)
    if "none" in args.signal_planting_strategy:
        with open(Path(foldername, "config_files", f"config_strategy_none_{args.fold}.yaml"), 'w') as yaml_outfile:
            yaml.dump({"nr_of_train_set_songs": nr_of_train_set_songs, "nr_of_test_set_song_recommendations": nr_of_test_set_song_recommendations}, yaml_outfile, default_flow_style=False)
        
        # all_playlists_manipulated = all_playlists

        # # TODO: just a test for empty playlist, remove again later
        collective_indices = np.random.choice(len(all_playlists), size=budget, replace=False)
        all_indices_not_in_collective = set(range(len(all_playlists))) - set(collective_indices)
        # collective = [all_playlists[i] for i in collective_indices]
        # for playlist in collective:
        #     playlist["tracks"] = []

        # use non-collective controlled duplicate playlists instead of considering the ones in the collective
        all_playlists_manipulated = [all_playlists[i] for i in all_indices_not_in_collective]
        for i, idx in enumerate(collective_indices):
            # duplicate one of the non-collective controlled playlists
            all_playlists_manipulated.append(copy.deepcopy(all_playlists_manipulated[i]))
            # overwrite the pid of the non-collective controlled playlists with the ones in the collective
            all_playlists_manipulated[-1]["pid"] = all_playlists[idx]["pid"]



    else:

        outfile_fp = Path(args.base_path, "submissions_and_results", f"2023_deezer_{args.model_name}", args.dataset_size, manipulated_dataset_directory, args.fold)
        outfile_fp.mkdir(parents=True, exist_ok=True)
        
        if "hybrid_optimum" in args.signal_planting_strategy:
            # hybrid strategy that is optimal dual placement strategy with position-specific placement for target song
            strategy_cooccurrence = "optimal_cooccurrence_top1_lambda1"
            if args.signal_planting_strategy == "hybrid_optimum":
                strategy_bigram = "optimal_bigram_top1"
            elif "hybrid_optimum_dynamicK" in args.signal_planting_strategy:
                dynamicK_parts = [part for part in parts if "dynamicK" in part]
                if dynamicK_parts:  # Check if dynamicK is in the strategy
                    dynamicK_str = dynamicK_parts[0]  # Take the first match
                    dynamicK_value = int(dynamicK_str.replace("dynamicK", ""))

                    strategy_bigram = f"optimal_bigram_top1_dynamicK{dynamicK_value}"
                    print(">>>>!!! running hybrid_optimum_dynamicK strategy with dynamicK_value:", dynamicK_value)
                else:
                    raise ValueError("dynamicK not in strategy, but should be")
            #pdb.set_trace()
            collective_with_signal, duplicate_playlists_to_insert, signals_planted_cooccurrence, collective_action_description_cooccurrence = plant_signal(known_training_data, unknown_training_data, track_id_by_track_uri, all_song_info, strategy_cooccurrence, budget, signal_track_ids, rng, outfile_fp, args.fold, collective=collective, plant_target_song=False, promotedifferentsongs=get_promotedifferentsongs, overshoot=overshoot, plantseed=plantseed, insertbefore=insertbefore, replaceseed=replaceseed, duplicateseed=duplicateseed, promoteexisting=promoteexisting)
            #pdb.set_trace()
            print(">>>done with cooccurrence strategy")
            print(f"Total GB of RAM used: {psutil.virtual_memory().used / (1024**3)}")
            print(">>>now running bigram strategy")

            # update known_training_data with the signals planted by the cooccurrence strategy
            known_training_data = []
            j=0
            for i, playlist in enumerate(all_playlists):
                if playlist["pid"] in known_training_data_playlist_pids:
                    if playlist["pid"] in collective_playlistpids:
                        known_training_data.append(collective_with_signal[j])
                        j+= 1
                    else:
                        known_training_data.append(all_playlists_only_track_ids[i])

            collective_with_signal, duplicate_playlists_to_insert, signals_planted, collective_action_description_bigram = plant_signal(known_training_data, unknown_training_data, track_id_by_track_uri, all_song_info, strategy_bigram, budget, signal_track_ids, rng, outfile_fp, args.fold, collective=collective, promotedifferentsongs=get_promotedifferentsongs, overshoot=overshoot, plantseed=plantseed, insertbefore=insertbefore, replaceseed=replaceseed, duplicateseed=duplicateseed, promoteexisting=promoteexisting)
            #pdb.set_trace()
            print(f"signals planted cooccurrence: {signals_planted_cooccurrence}")
            print(f"signals planted bigram: {signals_planted}")
            collective_action_description = "hybrid_optimum" + collective_action_description_cooccurrence + " >>>>>> running again now with bigram strategy <<<<<< " + collective_action_description_bigram
        else:
            # get famous songs by id
            if onefamoussong:
                famoussongs_as_track_id = track_id_by_track_uri[onefamoussong]
            else:
                famoussongs_as_track_id = ""
            
            if severalfamoussongs:
                severalfamoussongs_as_track_id = [track_id_by_track_uri[fam_song] for fam_song in severalfamoussongs]
            else:
                severalfamoussongs_as_track_id = []
            
            if "promoteexisting" in args.signal_planting_strategy:
                promoteexisting_as_track_id = track_id_by_track_uri[promoteexisting]
            else:
                promoteexisting_as_track_id = ""

            fp_fast_poisonbl = str(Path(foldername, "poisonbl_data", replication_folder_name))
            
            collective_with_signal, duplicate_playlists_to_insert, signals_planted, collective_action_description = plant_signal(known_training_data, unknown_training_data, track_id_by_track_uri, all_song_info, args.signal_planting_strategy, budget, signal_track_ids, rng, outfile_fp, args.fold, collective=collective, promotedifferentsongs=get_promotedifferentsongs, overshoot=overshoot, plantseed=plantseed, insertbefore=insertbefore, replaceseed=replaceseed, duplicateseed=duplicateseed, onefamoussong=famoussongs_as_track_id, severalfamoussongs=severalfamoussongs_as_track_id, promoteexisting_as_track_id=promoteexisting_as_track_id, fp_fast_poisonbl=fp_fast_poisonbl)
        
        print(f"Total GB of RAM used (after signals have been planted): {psutil.virtual_memory().used / (1024**3)}")

        # save all args as dict to yaml file in fp
        combined_dict = {**vars(args), "signal": signal, **signal_metadata, "collective_action_description": collective_action_description, "train_set_size": len(train_indices), "signals_planted": signals_planted, "nr_of_train_set_songs": nr_of_train_set_songs, "nr_of_test_set_song_recommendations": nr_of_test_set_song_recommendations, "onefamoussong": onefamoussong, "severalfamoussongs": severalfamoussongs}
        with open(Path(foldername, "config_files", f"config_strategy_{args.signal_planting_strategy}_{budget}_{args.fold}{args.collective_requirement}.yaml"), 'w') as yaml_outfile:
            yaml.dump(combined_dict, yaml_outfile, default_flow_style=False)

        all_playlists_manipulated = save_data_with_signal(collective_with_signal, collective_playlistpids, all_playlists, None, None, signal_track_ids, signal_metadata, all_song_info, track_id_by_track_uri, rng, duplicate_playlists_to_insert=duplicate_playlists_to_insert)

    #pdb.set_trace()
    np.save('%s/dataset_split/%s/train_indices' % (foldername, replication_folder_name), train_indices)
    np.save('%s/dataset_split/%s/val_indices' % (foldername, replication_folder_name), val_indices)
    np.save('%s/dataset_split/%s/test_indices' % (foldername, replication_folder_name), test_indices)


    ######## ######## ######## ######## ########
    ###### PROCESS MANIPULATED PLAYLISTS #######
    ######## ######## ######## ######## ########

    recount_planted_signals_in_final_dataset = 0
    recount_planted_signals_outside_collective = 0
    print("Original preprocessing on manipulated playlists", flush=True)
    for playlist in tqdm.tqdm(all_playlists_manipulated, disable=disable_progress_bar, file=sys.stdout):
        if "none" not in args.signal_planting_strategy:
            if playlist["pid"] not in collective_playlistpids:
                if signal in [track["track_uri"] for track in playlist["tracks"]]:
                    #print(f"  Expected signal {signal} not to be in playlist {playlist['pid']}, but it is in there, even though it's not part of the collective!!!")
                    recount_planted_signals_outside_collective += 1
            elif signal in [track["track_uri"] for track in playlist["tracks"]]:
                recount_planted_signals_in_final_dataset += 1
            else:
                print(f"Expected signal {signal} to be in playlist {playlist['pid']} (as it is part of the collective), but it is not in there")
                print("  playlist:", playlist)
        process_playlist(playlist)
    
    print(f"recount_planted_signals_in_final_dataset: {recount_planted_signals_in_final_dataset}")
    print(f"  recount_planted_signals_outside_collective: {recount_planted_signals_outside_collective}")

    seqfile = open('%s/playlists_seq.csv' % out_path, 'a', newline ='')
    with seqfile:
        write = csv.writer(seqfile)
        write.writerows(playlists_list)
    


    show_summary()


def show_summary():
    print()
    print("number of playlists", total_playlists)
    # assert total_playlists == n_playlists, f"Expected {n_playlists} playlists, but got {total_playlists} playlists"
    print("number of tracks", total_tracks)
    print("number of unique tracks", len(tracks))
    #assert len(tracks) == n_tracks, f"Expected {n_tracks} unique tracks, but got {len(tracks)} unique tracks"
    if len(tracks) != n_tracks:
        print(f"WARNING: assert error: Expected {n_tracks} unique tracks, but only got {len(tracks)} unique tracks")
    # assert len(tracks) <= n_tracks, f"Expected {n_tracks} unique tracks, but got {len(tracks)} unique tracks. playlist_track lil_matrix is initialized with n_tracks={n_tracks}, so does not work"
    print("number of unique albums", len(albums))
    print("number of unique artists", len(artists))
    print("number of unique titles", len(titles))
    print("number of playlists with descriptions", total_descriptions)
    print("number of unique normalized titles", len(ntitles))
    print("avg playlist length", float(total_tracks) / total_playlists)
    print()
    print("top playlist titles")
    for title, count in title_histogram.most_common(20):
        print("%7d %s" % (count, title))

    print()
    print("top tracks")
    for track, count in track_histogram.most_common(20):
        print("%7d %s" % (count, track))

    print()
    print("top artists")
    for artist, count in artist_histogram.most_common(20):
        print("%7d %s" % (count, artist))

    print()
    print("numedits histogram")
    for num_edits, count in num_edits_histogram.most_common(20):
        print("%7d %d" % (count, num_edits))

    print()
    print("last modified histogram")
    for ts, count in last_modified_histogram.most_common(20):
        print("%7d %s" % (count, to_date(ts)))

    print()
    print("playlist length histogram")
    for length, count in playlist_length_histogram.most_common(20):
        print("%7d %d" % (count, length))

    print()
    print("num followers histogram")
    for followers, count in num_followers_histogram.most_common(20):
        print("%7d %d" % (count, followers))


def normalize_name(name):
    name = name.lower()
    name = re.sub(r"[.,\/#!$%\^\*;:{}=\_`~()@]", " ", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name


def to_date(epoch):
    return datetime.datetime.fromtimestamp(epoch).strftime("%Y-%m-%d")


def process_playlist(playlist):
    global total_playlists, total_tracks, total_descriptions, unique_track_count, playlists_list

    total_playlists += 1
    # print playlist['playlist_id'], playlist['name']

    if "description" in playlist:
        total_descriptions += 1

    titles.add(playlist["name"])
    nname = normalize_name(playlist["name"])
    ntitles.add(nname)
    title_histogram[nname] += 1

    playlist_length_histogram[playlist["num_tracks"]] += 1
    last_modified_histogram[playlist["modified_at"]] += 1
    num_edits_histogram[playlist["num_edits"]] += 1
    num_followers_histogram[playlist["num_followers"]] += 1
    playlist_id = playlist["pid"]
    playlist_track_count = 0
    playlist_seq = []
    for track in playlist["tracks"]:
      full_name = track["track_uri"].lstrip("spotify:track:")
      if full_name not in tracks_info :
        del track["pos"]
        tracks_info[full_name] = track
        unique_track_count += 1
        tracks_info[full_name]["id"] = unique_track_count - 1
        tracks_info[full_name]["count"] = 1
      elif playlist_track[playlist_id, tracks_info[full_name]["id"]] != 0 :
        # remove tracks that are already earlier in the playlist
        continue
      else :
        tracks_info[full_name]["count"] += 1
      total_tracks += 1
      albums.add(track["album_uri"])
      tracks.add(track["track_uri"])
      artists.add(track["artist_uri"])
      artist_histogram[track["artist_name"]] += 1
      track_histogram[full_name] += 1
      track_id = tracks_info[full_name]["id"]
      playlist_track_count += 1
      playlist_track[playlist_id, track_id] = playlist_track_count
      playlist_seq.append(str(track_id))
    playlists_list.append(playlist_seq)


def process_info(_):
    pass

def process_album_artist( tracks_info, out_path):
    artist_songs = defaultdict(list)  # a dict where keys are artist ids and values are list of corresponding songs
    album_songs = defaultdict(list)  # a dict where keys are album ids and values are list of corresponding songs
    song_album = np.zeros(n_tracks)  # a 1-D array where the index is the track id and the value is the album id
    song_artist = np.zeros(n_tracks)  # a 1-D array where the index is the track id and the value is the artist id
    album_ids = {}  # a dict where keys are album names and values are album ids
    artist_ids = {}  # a dict where keys are artist names and values are artist ids
    album_names = []  # a list where indices are album ids and values are album names
    artist_names = []  # a list where indices are artist ids and values are album names
    print("Processing albums and artists.")
    for d in tqdm.tqdm(tracks_info.values(), disable=disable_progress_bar, file=sys.stdout):
        album_name = "%s by %s" % (d['album_name'], d['artist_name'])
        artist_name = d['artist_name']
        if album_name not in album_ids:
            album_id = len(album_names)
            album_ids[album_name] = album_id
            album_names.append(album_name)
        else:
            album_id = album_ids[album_name]
        song_album[d['id']] = album_id

        if artist_name not in artist_ids:
            artist_id = len(artist_names)
            artist_ids[artist_name] = artist_id
            artist_names.append(artist_name)
        else:
            artist_id = artist_ids[artist_name]
        song_artist[d['id']] = artist_id
        album_songs[album_id].append(d['id'])
        artist_songs[artist_id].append(d['id'])

    np.save('%s/song_album' % out_path, song_album)
    np.save('%s/song_artist' % out_path, song_artist)
    with open("%s/album_ids.pkl" % out_path, 'wb+') as f:
      pickle.dump(album_ids, f, protocol=4)

    with open("%s/artist_ids.pkl" % out_path, 'wb+') as f:
      pickle.dump(artist_ids, f, protocol=4)

    with open("%s/artist_songs.pkl" % out_path, 'wb+') as f:
      pickle.dump(artist_songs, f, protocol=4)

    with open("%s/album_songs.pkl" % out_path, 'wb+') as f:
      pickle.dump(album_songs, f, protocol=4)

    with open("%s/artist_names.pkl" % out_path, 'wb+') as f:
      pickle.dump(artist_names, f, protocol=4)

    with open("%s/album_names.pkl" % out_path, 'wb+') as f:
      pickle.dump(album_names, f, protocol=4)
    return


if __name__ == "__main__":
    print("deezer beginning of main")
    start_time = time.perf_counter()
    parser = argparse.ArgumentParser()
    parser.add_argument("--mpd_path", type=str, required=False, default="../MPD/data",
                             help = "Path to MPD")
    parser.add_argument("--out_path", type=str, required=False, default="resources/data/rta_input",
                             help = "Path to rta input")

    #parser.add_argument('-c', '--config_filename', type=str, required=True, help='config file name.')
    parser.add_argument('-strategy', '--signal_planting_strategy', type=str, required=True)
    parser.add_argument('-b', '--budget', type=int, help='the number of signals to plant.')
    parser.add_argument('-s', '--seed', type=int, required=True, help='seed for random number generation.')
    parser.add_argument('-collr', '--collective_requirement', type=str, required=False, default="", help='empty string if collective is random subset or track uri if every collective playlist must contain this track.')
    #parser.add_argument('-rss', '--recommendation_strategy_sampling', type=str, required=True, help='How the next song should be recommended ("proportional" or "random").')
    #parser.add_argument('-rsk', '--recommendation_strategy_k', type=str, required=True, help='Top-k songs to consider for each recommendation. Must be either an integer as string or "inf").')

    #parser.add_argument('--threads', type=int, required=True)
    parser.add_argument('--base_path', type=str, required=True)
    parser.add_argument('--base_path_data', type=str, required=True)
    parser.add_argument('--dataset_name', type=str, required=True)
    parser.add_argument('--dataset_size', type=str, required=True)
    #parser.add_argument('--manipulated_dataset_directory', type=str, required=True)
    parser.add_argument('--fold', type=str, required=True)
    parser.add_argument('--challenge_dataset_name', type=str, required=True)
    parser.add_argument('--outfile', type=str, required=True)
    parser.add_argument('--model_name', type=str, required=True)
    
    args = parser.parse_args()


    if "none" in args.signal_planting_strategy:
        manipulated_dataset_directory = args.signal_planting_strategy
    else:
        manipulated_dataset_directory = f"signal_planting_strategy_{args.signal_planting_strategy}_budget_{args.budget}{args.collective_requirement}"


    mpd_path = str(Path(args.base_path_data, "data", args.dataset_name, args.dataset_size, "_original_dataset"))
    print(f"loading mpd from {mpd_path}")

    out_path = Path(args.base_path_data, "recommender_system/2023_deezer_transformers/resources/data/rta_input", args.dataset_name, args.dataset_size, manipulated_dataset_directory, args.fold)
    out_path.mkdir(parents=True, exist_ok=True)
    out_path = str(out_path)
    
    models_path = Path(args.base_path_data, "recommender_system/2023_deezer_transformers/resources/models", args.dataset_name, args.dataset_size, manipulated_dataset_directory, args.fold)
    models_path.mkdir(parents=True, exist_ok=True)

    dataset_split_path = Path(args.base_path_data, "recommender_system/2023_deezer_transformers/resources/data/dataset_split", args.dataset_name, args.dataset_size, manipulated_dataset_directory, args.fold)
    dataset_split_path.mkdir(parents=True, exist_ok=True)
    
    data_manager_path = str(Path(args.dataset_name, args.dataset_size, manipulated_dataset_directory, args.fold))

    foldername = str(Path(args.base_path_data, "recommender_system/2023_deezer_transformers/resources/data/"))

    #n_tracks = 2262292
    n_tracks = get_nr_of_unique_tracks(args.signal_planting_strategy)
    playlist_track = lil_matrix((n_playlists, n_tracks), dtype=np.int32) # to build interaction matrix of binary value



    #os.makedirs(args.out_path, exist_ok=True)
    #os.makedirs("resources/models", exist_ok=True)
    process_mpd(mpd_path, out_path, args, data_manager_path, foldername)
    #pdb.set_trace()
    print(f"Total GB of RAM used (after process_mpd): {psutil.virtual_memory().used / (1024**3)}")
    save_npz('%s/playlist_track.npz' % out_path, playlist_track.tocsr(False))
    with open('%s/tracks_info.json' % out_path, 'w') as fp:
      json.dump(tracks_info, fp, indent=4)
    process_album_artist(tracks_info, out_path)
    print(f"Total GB of RAM used (after process_album_artist): {psutil.virtual_memory().used / (1024**3)}")

    print("end deezer :):)")
    print("  >>>Finished preprocessing in {:.4f} seconds".format(time.perf_counter() - start_time))