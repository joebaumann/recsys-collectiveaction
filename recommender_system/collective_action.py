import os
import subprocess
import shutil
import sys
import time
import numpy as np
import tqdm
from collections import defaultdict, Counter
import copy
import heapq
import math
import pdb
import psutil
import utils
import csv
from pathlib import Path
from utils import get_overshoot_int, get_percentile, get_targetx, get_nr_of_unique_tracks, disable_progress_bar
import pickle
from itertools import cycle
import pandas as pd
import re
from bs4 import BeautifulSoup
from difflib import SequenceMatcher
import requests
import seaborn as sns
import matplotlib.pyplot as plt
import json
from datetime import datetime
from requests.exceptions import ProxyError

@utils.timer
def update_playlists_for_upcoming_seed_songs(playlists, result, song):
    cnt_test=0
    for s, pls in playlists.items():
        to_remove = []
        for (ind, pl) in pls:
            # check if the playlist is newly taken for the result
            if (ind, pl) in result[song]:
                to_remove.append((ind,pl))
                cnt_test+=1
                #print(f"playlist with ind={ind} has been removed from candidate playlists for candidate seed song {s}")
        # Remove the tuples from the playlists
        for (ind, pl) in to_remove:
            playlists[s].remove((ind, pl))
    return playlists

@utils.timer
def maximize_success_bigram_greedy(utilities, collective, budget, sampling, top_k, already_used_playlists, plantseed):
    print("start of maximize_success_bigram_greedy")
    # Sort the songs by success-to-cost ratio in descending order
    # TODO: for large budgets, sorting is inefficient, make faster
    print("start of sorting", flush=True)
    print(f"len(utilities): {len(utilities)}", flush=True)
    start_time = time.perf_counter()
    songs = sorted(utilities.keys(), key=lambda s: utilities[s][2], reverse=True)
    print(f"   sorting done in {time.perf_counter()-start_time:.4f} seconds", flush=True)
    # TODO: make this more efficient
    playlists = {song: [(index, playlist) for index, playlist in enumerate(collective) if song in playlist and (index,tuple(playlist)) not in already_used_playlists] for song in songs}
    print(f"nr of remaining playlists from collective: {len([(index, playlist) for index, playlist in enumerate(collective) if (index,tuple(playlist)) not in already_used_playlists])}")

    total_success = 0
    total_costs = 0
    weight_increase_of_previous_results = []
    nr_of_seed_song_candidates = len(songs)

    # initialize result dict that stores s_s and the corresponding playlists in which s* should be planted after s_s
    result = defaultdict(list)
    for i, song in enumerate(songs):
        cost, success, ratio, tot_freq, tot_freq_of_top_k = utilities[song]
        #print(f"\n--seed song candidate {i}/{len(songs)}-- [cost:{cost},success:{success},ratio:{ratio},tot_freq:{tot_freq},tot_freq_of_top_k:{tot_freq_of_top_k}]")

        #### #### #### #### #### ####
        ### check weight increase ###
        #### #### #### #### #### ####
        # if the strategy is proportional, check if the utility ratio for increasing the weight for one of the seed contexts in the result is higher than the ratio of the current seed song
        # increase the weight of the seed song with the highest ratio as long as it is higher than the ratio of the current seed song
        if sampling == "proportional" and top_k > 1:
            while len(weight_increase_of_previous_results) > 0 and max([r[2] for r in weight_increase_of_previous_results]) > ratio:

                weight_increase_of_previous_results.sort(key=lambda x: x[2], reverse=True)
                existing_result_with_max_ratio = weight_increase_of_previous_results.pop(0)
                prev_song, prev_cost, prev_success_of_weight_increase_by_one, prev_success, prev_tot_freq, prev_tot_freq_of_top_k = existing_result_with_max_ratio

                if 1 <= budget and len(playlists[prev_song]) >= 1:
                    print(existing_result_with_max_ratio," > ", ratio)
                    # increase the weight of existing seed song with the highest ratio
                    #pdb.set_trace()
                    
                    
                    # Include the song in the solution if its cost are below the budget and if there are still enough playlists left that contain the song
                    total_success += prev_success_of_weight_increase_by_one
                    total_costs += 1
                    budget -= 1
                    tqdm.tqdm.write(f" TAKEN :) WEIGHT INCREASE: remaining budget: {budget}, newly added success: {prev_success_of_weight_increase_by_one}, total success: {total_success}, total costs: {total_costs}, result: {len(result.keys())} with {sum(len(v) for v in result.values())} playlists")
                    # decide which playlist to use
                    # If there are still more than 'cost' playlists left, remove the smallest ones
                    if len(playlists[song]) > 1:
                        result[prev_song] = sorted(copy.deepcopy(playlists[prev_song]), key=lambda p: len(p))[:1]
                    else:
                        result[prev_song] = copy.deepcopy(playlists[prev_song])
                    
                    result[prev_song].extend(playlists[prev_song])

                    # update the candidate playlists of the upcoming seed songs
                    playlists = update_playlists_for_upcoming_seed_songs(playlists, result, prev_song)

                    # check if there is room for increasing the weight
                    if (prev_tot_freq-prev_cost) > 0:
                        tot_freq_of_top_k_after_addition = prev_tot_freq_of_top_k + (prev_tot_freq_of_top_k-prev_cost) / (prev_tot_freq-prev_cost)
                        success_of_weight_increase_by_one = (prev_cost+1) * (1 / ( tot_freq_of_top_k_after_addition / prev_tot_freq )) - prev_success
                        #pdb.set_trace()
                        weight_increase_of_previous_results.append([prev_song, prev_cost+1, success_of_weight_increase_by_one, prev_success+success_of_weight_increase_by_one, prev_tot_freq, tot_freq_of_top_k_after_addition])
                
                if budget == 0:
                    break


        #### #### #### #### #### ####
        #### check new candidate ####
        #### #### #### #### #### ####
        if cost <= budget and (plantseed or len(playlists[song]) >= cost):
            # Include the song in the solution if its cost are below the budget and if there are still enough playlists left that contain the song
            total_success += success
            total_costs += cost
            budget -= cost
            tqdm.tqdm.write(f" TAKEN :) [cost:{cost},success:{success},ratio:{ratio},tot_freq:{tot_freq},tot_freq_of_top_k:{tot_freq_of_top_k}] remaining budget: {budget}  ---  looking at seed song candidate {i}/{nr_of_seed_song_candidates}\n")
            # decide which playlist to use
            j = i + 1
            # if len(playlists[song]) == cost, we need to choose all of those playlists
            while j < len(songs) and len(playlists[song]) > cost:
                if utilities[songs[j]][0] <= budget:
                # check if the remaining budget is sufficient for the next seed song to be considered
                    for (index,playlist) in playlists[song]:
                        # go through the candidate playlists
                        # check if this playlist would be used by upcoming seed songs
                        # this will be the upcoming seed song, thus, check if the playlist is used for this seed song
                        playlists_indices_of_next_seed_song_candiate = [ind for ind, _ in playlists[songs[j]]]
                        # check which playlist (of those associated with the upcoming seed song) containing the song use this seed song
                        if index in playlists_indices_of_next_seed_song_candiate:
                            # this playlist is used for the upcoming seed song, thus, do not use for this seed song
                            playlists[song].remove((index,playlist))
                            #print(f"  playlist with ind={index} has been removed from candidate playlists for candidate seed song {cost}, since it is also used for the {j}-next upcoming seed song {songs[j]}")
                        # check if there are still enough playlists left
                        if len(playlists[song]) <= cost:
                            break
                j += 1
            # If there are still more than 'cost' playlists left, remove the smallest ones
            if len(playlists[song]) > cost:
                print(f" len(playlists[song])={len(playlists[song])} > cost={cost}")
                result[song] = sorted(copy.deepcopy(playlists[song]), key=lambda p: len(p))[:cost]
            elif len(playlists[song]) < cost:
                print(f" len(playlists[song])={len(playlists[song])} < cost={cost}")
                # If there are less than 'cost' playlists left, add all of them
                result[song] = copy.deepcopy(playlists[song])
                # randomly select other playlists to use
                nr_of_playlists_available_including_duplicates = len(playlists)
                rng = np.random.default_rng(seed=42)
                while len(result[song]) < cost:
                    # randomly chose a playlist from playlists.values(), check if it is already in result[song], otherwise add it
                    random_playlist = tuple(rng.choice([tpl for sublist in playlists.values() for tpl in sublist]))
                    if random_playlist not in result[song]:
                        result[song].append(copy.deepcopy(random_playlist))
                        print(f" adding a random playlist to the result, since there are not enough playlists left to fill the result")
                        print(f" now there are {len(result[song])} playlists in the result")
            else:
                result[song] = copy.deepcopy(playlists[song])
                        
            # update the candidate playlists of the upcoming seed songs
            playlists = update_playlists_for_upcoming_seed_songs(playlists, result, song)

            if sampling == "proportional" and top_k > 1:
                # check if there is room for increasing the weight
                if (tot_freq-cost) > 0:
                    tot_freq_of_top_k_after_addition = tot_freq_of_top_k + (tot_freq_of_top_k-cost) / (tot_freq-cost)
                    success_of_weight_increase_by_one = (cost+1) * (1 / ( (tot_freq_of_top_k_after_addition) / tot_freq )) - success
                    #pdb.set_trace()
                    weight_increase_of_previous_results.append([song, cost+1, success_of_weight_increase_by_one, success, tot_freq, tot_freq_of_top_k_after_addition])

        
        if budget == 0:
            break
    
    #print(f"the total costs are {total_costs}, the total success is {total_success}")
    #print(f"{len(result.keys())} seed songs are part of the optimal solution, the remaining budget is {budget}")
    #assert budget == len(collective)-total_costs, "the remaining budget should be equal to the size of the collective minus the total costs used for the optimal solution"
    if budget != len(collective)-total_costs:
        print(f"\n !!! the remaining budget should be equal to the size of the collective minus the total costs used for the optimal solution \n\t budget is {budget} / collective has size {len(collective)} / total costs are {total_costs} !!!\n")

    # go through the optimal solution and save as playlist:song, assert that every playlist is only used once
    playlist_to_song = defaultdict(list)
    for result_song, result_playlists in result.items():
        for (index, playlist) in result_playlists:
            playlist_to_song[(index, tuple(playlist))].append(result_song)
    
    # Check if any list has more than 1 items
    for key, songs in playlist_to_song.items():
        if len(songs) > 1:
            print(f"Error: the target is inserted along with {len(songs)} into the playlist {key}, which is more then the 1 seed allowed for the bigram strategy.")

    return playlist_to_song, budget

@utils.timer
def remove_already_used_playlists(result, remaining_options, lambda_value, to_remove=[]):
    
    # remove already used playlists
    for res in result.values():
        for (ind, pl) in res:
            if (ind, pl) in remaining_options:
                # remove an already taken playlist from this songs candidates
                #if index_and_playlist not in to_remove:
                to_remove.append((ind, tuple(pl)))
    # Count occurrences of each tuple in to_remove
    playlist_counts = Counter(to_remove)
    # Filter out tuples that occur >= lambda_value times
    filtered_to_remove = [tup for tup in to_remove if playlist_counts[tup] >= lambda_value]
    # Remove the filtered playlists
    for (ind, pl) in filtered_to_remove:
        if (ind, pl) in remaining_options:
            remaining_options.remove((ind, pl))
        elif (ind, tuple(pl)) in remaining_options:
            remaining_options.remove((ind, tuple(pl)))
        elif (ind, list(pl)) in remaining_options:
            remaining_options.remove((ind, list(pl)))
    
    return remaining_options


def select_one_of_most_freq_songs(candidates, frequencies, n, rng):
    candidate_occurences = [(t, frequencies[t]) for t in candidates]
    #print("test max freq: ", max(frequencies.values()))
    # get the most famous song among all those present in this playlist
    most_famous_tracks = [track for (track, count) in heapq.nlargest(n, candidate_occurences, key=lambda x: x[1])]
    #print(f"most_famous_tracks: {most_famous_tracks}")
    sample_of_most_famous_tracks = rng.choice(most_famous_tracks)
    #print(f"sample_of_most_famous_tracks: {sample_of_most_famous_tracks}, with frequency {frequencies[sample_of_most_famous_tracks]}")
    return sample_of_most_famous_tracks

def get_x_most_frequent_songs(frequencies, x_most_frequent_songs_to_consider=1):
    x_most_frequent_songs = [song for song, count in frequencies.most_common(x_most_frequent_songs_to_consider)]
    #print(f"the x_most_frequent_songs_to_consider most common songs in frequencies are: {x_most_frequent_songs}")
    return x_most_frequent_songs

def select_one_of_x_most_freq_songs(candidates, frequencies, n, rng, x_most_frequent_songs):
    # select the x_most_frequent_songs_to_consider most common songs in frequencies
    candidate_occurences = [(t, frequencies[t]) for t in candidates if t in x_most_frequent_songs]
    #print("candidate_occurences: ", candidate_occurences)
    if len(candidate_occurences) == 0:
        # no songs in candidates are among the x_most_frequent_songs_to_consider most common songs in frequencies, so select one of the most common songs in frequencies
        print("no songs in candidates are among the x_most_frequent_songs_to_consider most common songs in frequencies, so select one of the most common songs in frequencies")
        return None
    #print("test max freq: ", max(frequencies.values()))
    # get the most famous song among all those present in this playlist
    most_famous_tracks = [track for (track, count) in heapq.nlargest(n, candidate_occurences, key=lambda x: x[1])]
    #print(f"most_famous_tracks: {most_famous_tracks}")
    sample_of_most_famous_tracks = rng.choice(most_famous_tracks)
    #print(f"sample_of_most_famous_tracks: {sample_of_most_famous_tracks}, with frequency {frequencies[sample_of_most_famous_tracks]}")
    return sample_of_most_famous_tracks


def get_song_freq_with_max_percentile(frequencies, percentile):
    # Convert the frequencies to a list of frequency values
    freq_values = list(frequencies.values())
    # Calculate the threshold frequency for the given percentile
    threshold = np.percentile(freq_values, percentile)
    # Filter the Counter to include only those items with frequency <= threshold
    frequencies_with_max_percentile = Counter({track: freq for track, freq in frequencies.items() if freq <= threshold})
    print(f"the new frequencies_with_max_percentile has {len(frequencies_with_max_percentile)} songs, which corresponds to {len(frequencies_with_max_percentile)/len(frequencies)} of the original frequencies, which had a total of {len(frequencies)} songs. [note: percentile threshold is {threshold}]")
    return frequencies_with_max_percentile


def select_song_with_rank_closest_to_targeted_percentile(candidates, percentile_df, target_percentile, rng):
    # Filter the DataFrame to only include candidates
    candidate_df = percentile_df[percentile_df.index.isin(candidates)]

    print(f"  heyheytest: shape of candidate_df is {candidate_df.shape}, shape of percentile_df is {percentile_df.shape}")
    
    print(f"  heyheyheytest: candidate_df is {candidate_df}")

    # Find the candidate whose percentile rank is closest to the targeted percentile
    closest_song = candidate_df.iloc[(candidate_df['Percentile Rank'] - target_percentile).abs().argsort()[:1]].index[0]
    # TODO: there are quite a few songs with the same percentile rank. use rank instead (e.g., min/max depending on dist from target), at least when there are multiple songs with the same percentile rank
    print(f"  heyheytest: closest_song is {closest_song}, with percentile rank {candidate_df.loc[closest_song]['Percentile Rank']} and target percentile {target_percentile}")

    return closest_song

def insert_songs_in_playlist(playlist, index, songs, rng, insertbefore, replaceseed, duplicateseed, duplicate_playlists_to_insert, probabilities=None):
    # insert the songs in reverse order, so that the indices do not change
    for i, song in enumerate(reversed(songs)):
        if probabilities is not None:
            proba = reversed(probabilities)[i]
            # continue with probability 1-proba
            if rng.random() > proba:
                # do not insert this song
                continue

        if replaceseed:
            assert len(songs) == 1, "replaceseed can only be used if there is only one song to insert, TODO: implement functionality to replace one seed song with multiple target songs"
            #pdb.set_trace()
            # replace the seed song
            # first, insert signal after seed song
            playlist.insert(index, song)
            # then, remove seed song from playlist
            del playlist[index-1]
        elif duplicateseed:
            assert len(songs) == 1, "duplicateseed can only be used if there is only one song to insert, TODO: implement functionality to replace one seed song with multiple target songs"
            #pdb.set_trace()
            # duplicate the playlist, and replace the seed song in the duplicate, leave the original unchanged
            # first, duplicate the playlist and add it to a list to later use it to replace a random existing playlist that is not controlled by the collective
            duplicate_playlists_to_insert.append(copy.deepcopy(playlist))
            # next, replace the seed song in the original playlist
            # first, insert signal after seed song
            playlist.insert(index, song)
            # then, remove seed song from playlist
            del playlist[index-1]
        else:
            if insertbefore:
                # insert signal before seed song
                playlist.insert(index-1, song)
                #print(f"inserted {song} BEFORE, i.e., at index {index-1}")
            else:
                # insert signal after seed song
                playlist.insert(index, song)
                #print(f"inserted {song} AFTER, i.e., at index {index}")

    return duplicate_playlists_to_insert


def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()



def get_spotify_streams(song_uri, track_name, max_retries=5, initial_wait=60):
    url = f"https://open.spotify.com/track/{song_uri}"
    attempt = 0
    while attempt < max_retries:
        try:
            response = requests.get(url)
            # If the request is successful, break out of the loop
            if response.status_code == 200:
                # Process your response here
                url, closest_stream_number = process_spotify_stream_url_requests(song_uri, track_name, url, response)
                return url, closest_stream_number  # Assuming closest_stream_number is determined elsewhere in your code
            else:
                print(f"Request failed with status code: {response.status_code}")
                return url, None
        except ProxyError as e:
            print(f"Caught proxy error: {e}. Attempt {attempt+1} of {max_retries}. Waiting {initial_wait} seconds before retrying.")
            time.sleep(initial_wait)
            # Increment the attempt counter and double the wait time for the next retry
            attempt += 1
            initial_wait *= 2  # Double the wait time for the next attempt

    print("Max retries exceeded. Giving up.")
    return url, None


def process_spotify_stream_url_requests(song_uri, track_name, url, response):
    soup = BeautifulSoup(response.text, 'html.parser')

    #extracting the number of streams

    # first extract the last row in soup
    
    # Convert the soup object to a string
    soup_str = str(soup)

    #print("soup_str:", soup_str)
    
    # Regular expression to find numbers and potential track names within the specified HTML structure
    pattern_streams = r'data-encore-id="type">([\d,]+)</span>'
    pattern_tracks = r'data-encore-id="type">(.*?)</span>'

    # Regular expression to validate a properly formatted number with commas
    valid_number_pattern = r'^\d{1,3}(,\d{3})*$'

    # Find all occurrences of the patterns
    stream_matches = re.finditer(pattern_streams, soup_str)
    track_matches = re.finditer(pattern_tracks, soup_str)

    # Extract numbers and their indices
    numbers = []
    number_indices = []
    for match in stream_matches:
        number = match.group(1)
        # Check if the number is properly formatted
        if re.match(valid_number_pattern, number):
            numbers.append(number)
            number_indices.append(match.start(1))
    
    if numbers == []:
        print("Could not find streams for track:", track_name)
        #pdb.set_trace()


    # Extract potential track names and their indices
    track_candidates = []
    for match in track_matches:
        track_candidates.append((match.group(1), match.start(1)))

    # Filter track candidates based on similarity
    track_indices = []
    while track_indices == []:
        for candidate, index in track_candidates:
            if track_name in candidate or similar(candidate, track_name) > 0.4:  # Adjust similarity threshold as needed
                track_indices.append(index)
        # remove last character from track_name and try again
        track_name = track_name[:-1]
        if " " not in track_name:
            # keep at least one word from the title
            break
    
    if track_indices == []:
        print("Could not find track name:", track_name)
        #pdb.set_trace()

    # Find the stream number with the minimal distance following each track name occurrence
    closest_stream_number = None
    min_distance = float('inf')
    for track_index in track_indices:
        # Filter numbers that appear after the track name
        following_numbers = [(num, abs(num - track_index)) for num in number_indices if num > track_index]
        if following_numbers:
            # Find the closest number index to the track name
            closest_number_index, distance = min(following_numbers, key=lambda x: x[1])
            # Update if this is the closest stream number so far
            if distance < min_distance:
                min_distance = distance
                for number, index in zip(numbers, number_indices):
                    if index == closest_number_index:
                        # replace commas with empty string and make int
                        closest_stream_number = int(number.replace(",", ""))
                        break
    
    if closest_stream_number is None:
        print("Could not find stream number for track:", track_name)
        #pdb.set_trace()

    return url, closest_stream_number


def get_song_with_lowest_artist_popularity(playlist, all_song_info, artist_popularities, rng):
    if artist_popularities is None:
        # load artist_popularities
        filepath = "/home/jbaumann/spotify-algo-collective-action/RecSysChallengeSolutions/spotify_scraper/artist_popularities.json"

        if os.path.exists(filepath):
            with open(filepath, "r") as f:
                artist_popularities = json.load(f)
            print(f"Loaded {filepath} from disk")
        else:
            print(f"Could not find {filepath} on disk")
            pdb.set_trace()
    
    song_with_lowest_artist_popularity = None
    least_popular = np.inf
    for track_id in playlist:
        track_uri = all_song_info[track_id]["track_uri"].replace("spotify:track:", "")
        artist_uri = all_song_info[track_id]["artist_uri"].replace("spotify:artist:", "")
        track_name = all_song_info[track_id]["track_name"]
        artist_popularity = artist_popularities.get(artist_uri, None)
        if artist_popularity is None:
            # TODO: make API call to get artist scrape the streams for this song
            print(f"artitst popularity for artists with uri {artist_uri} not available [track_uri: {track_uri}, track_name: {track_name}, track_id: {track_id}]")
        if artist_popularity is not None:
            if artist_popularity < least_popular:
                least_popular = artist_popularity
                song_with_lowest_artist_popularity = track_id
    if song_with_lowest_artist_popularity is None:
        print("Could not find artist popularity for any song in the playlist")
        # randomly select a song from the playlist
        song_with_lowest_artist_popularity = rng.choice(playlist)

    return artist_popularities, song_with_lowest_artist_popularity


def get_song_with_fewest_streams(playlist, all_song_info, scraped_streams, rng):
    newly_scraped_streams = {}
    filepath_aggregated = "/home/jbaumann/spotify-algo-collective-action/RecSysChallengeSolutions/spotify_scraper/scraped_streams_aggregated.csv"
    if scraped_streams is None:
        # load scraped_streams
        if os.path.exists(filepath_aggregated):
            with open(filepath_aggregated, "r") as f:
                # load csv file into pd dataframe
                scraped_streams = pd.read_csv(f)
            print(f"Loaded scraped_streams.csv from disk")
            # transform df into dict
            scraped_streams = scraped_streams.set_index('track_uri')['streams'].to_dict()
        else:
            scraped_streams = {}
            print(f"Could not find scraped_streams.csv on disk")
    least_streamed_song = None
    least_streams = np.inf
    for track_id in playlist:
        track_uri = all_song_info[track_id]["track_uri"].replace("spotify:track:", "")
        track_name = all_song_info[track_id]["track_name"]
        stream_count = scraped_streams.get(track_uri, None)
        if stream_count is None:
            # scrape the streams for this song
            url, stream_count = get_spotify_streams(track_uri, track_name)
            print("scraped streams for track_id:", track_id, "and track_uri:", track_uri, "with stream_count:", stream_count, "and url:", url, " and track_name:", track_name)
            scraped_streams[track_uri] = stream_count
            newly_scraped_streams[track_uri] = stream_count
        else:
            print("already scraped streams for track_id:", track_id, "and track_uri:", track_uri, "with stream_count:", stream_count, "and track_name:", track_name)
        if stream_count is not None:
            try:
                # Attempt to convert stream_count to float first, then to int if necessary
                stream_count_float = float(stream_count)
                if stream_count_float < least_streams:
                    least_streams = int(stream_count_float)  # Convert to int for comparison and assignment
                    least_streamed_song = track_id
            except Exception as e:
                # Catch any exception that occurs during conversion or comparison
                print(f"Error processing stream count for track_id: {track_id}, track_uri: {track_uri}. Error: {e}")
    if least_streamed_song is None:
        print("Could not scrape streams for any song in the playlist")
        # randomly select a song from the playlist
        least_streamed_song = rng.choice(playlist)
    # save newly_scraped_streams to disk
    # convert dict to df
    # append to existing csv if exists
    if os.path.exists(filepath_aggregated):
        with open(filepath_aggregated, "a") as f:
            # append all from newly_scraped_streams_df
            for track_uri, stream_count in newly_scraped_streams.items():
                f.write(f"{track_uri},{stream_count}\n")
    else:
        # create new csv file
        with open(filepath_aggregated, "w") as f:
            # append all from newly_scraped_streams_df
            for track_uri, stream_count in newly_scraped_streams.items():
                f.write(f"{track_uri},{stream_count}\n")
    
    # newly_scraped_streams_df = pd.DataFrame.from_dict(newly_scraped_streams, orient='index')
    # # save as new csv file
    # start_date = datetime.now().strftime("%Y %m %d %H:%M:%S")
    # sanitized_date = start_date.replace(' ', '_').replace(':', '_')
    # newly_scraped_streams_df.to_csv(f"{filepath_aggregated}_NEW_{sanitized_date}", index=False)
    
    return scraped_streams, least_streamed_song


@utils.timer
def plant_signal(known_training_data, unknown_training_data, track_id_by_track_uri, all_song_info, signal_planting_strategy, budget, signals, rng, fp, fold=None, collective=None, plant_target_song=True, promotedifferentsongs=False, overshoot=0, plantseed=False, insertbefore=False, replaceseed=False, duplicateseed=False, onefamoussong=None, severalfamoussongs=[], promoteexisting_as_track_id=None, fp_fast_poisonbl=""):
    print("start of plant_signal")
    # make sure that the original known_training_data is not modified to avoid interference with other folds
    #known_training_data = copy.deepcopy(known_training_data)
    print("done with deepcopy")

    print(f" heyhey onefamoussong: {onefamoussong}")
    print(f" heyhey severalfamoussongs: {severalfamoussongs}")

    cyclic_signals_iterator = cycle(signals)
    seed_song_signals = {}

    def get_signals(seed_song=None):
        if seed_song is not None:
            if seed_song in seed_song_signals:
                # make sure to use the same signal if this seed_song has been used before
                return seed_song_signals[seed_song]
        if promotedifferentsongs:
            signal = [next(cyclic_signals_iterator)]
        else:
            signal = signals
        if seed_song is not None:
            # save the signal for this seed_song so that it can be used again
            seed_song_signals[seed_song] = signal
        return signal

    # choose a random subset of known_training_data of size budget

    if collective is None:
        # TODO: remove from here
        # TODO: use rng and budget as seed
        collective_indices = np.random.choice(len(known_training_data), size=budget, replace=False)
        collective = [known_training_data[i] for i in collective_indices]

    signals_planted = 0
    duplicate_playlists_to_insert = []

    song_occurences_in_train = Counter([track for playlist in known_training_data for track in playlist])
    song_occurences_in_train_INCLUDING_UNKNOWN = Counter([track for playlist in known_training_data+unknown_training_data for track in playlist])
    song_occurences_in_collective = Counter([track for playlist in collective for track in playlist])

    # TODO: make sure to comment out if there's no need to save to disk
    save_true_training_data_song_freqs_to_disk = True
    #save_true_training_data_song_freqs_to_disk = False
    if save_true_training_data_song_freqs_to_disk:
        # save true song occurences of songs in collective to disk
        filepath = Path(f"/fast/jbaumann/spotify-algo-collective-action/RecSysChallengeSolutions/2023_deezer_transformers/resources/data/training_data_frequencies/{signal_planting_strategy}_budget_{budget}/{fold}")
        filepath.mkdir(parents=True, exist_ok=True)
        with open(f"{filepath}/true_occurences_of_songs_in_collective.pkl", "wb") as f:
            true_song_occurrences_as_song_uris = {all_song_info[song_id]["track_uri"]: count for song_id, count in dict(song_occurences_in_train_INCLUDING_UNKNOWN).items()}
            pickle.dump(true_song_occurrences_as_song_uris, f)

        # save song_occurences_in_collective to disk
        with open(f"{filepath}/song_occurences_in_collective.pkl", "wb") as f:
            song_occurrences_as_song_uris = {all_song_info[song_id]["track_uri"]: count for song_id, count in dict(song_occurences_in_collective).items()}
            pickle.dump(song_occurrences_as_song_uris, f)

        # save song_occurences_in_train to disk
        with open(f"{filepath}/song_occurences_in_train.pkl", "wb") as f:
            train_song_occurrences_as_song_uris = {all_song_info[song_id]["track_uri"]: count for song_id, count in dict(song_occurences_in_train).items()}
            pickle.dump(train_song_occurrences_as_song_uris, f)


    # plot CDF

    # Extract data from the Counter object
    songs, counts = zip(*song_occurences_in_train.items())
    total_occurrences = sum(counts)

    # Create a DataFrame
    data = pd.DataFrame({'Song': songs, 'Count': counts}).sort_values('Count')
    data['Cumulative Percentage'] = data['Count'].cumsum() / total_occurrences * 100

    # Creating the cumulative distribution plot
    plt.figure(figsize=(10, 6))
    sns.lineplot(x='Song', y='Cumulative Percentage', data=data)
    plt.xticks(rotation=90)
    plt.title('Cumulative Distribution of Song Occurrences')
    plt.xlabel('Songs')
    plt.ylabel('Cumulative Percentage')
    plt.tight_layout()
    plt.savefig(f'{str(fp)}/cdf_song_occurrences.pdf', bbox_inches='tight')
    plt.close()

    # Extract data from the Counter object
    songs, counts = zip(*song_occurences_in_train_INCLUDING_UNKNOWN.items())
    total_occurrences = sum(counts)

    # Create a DataFrame
    data = pd.DataFrame({'Song': songs, 'Count': counts}).sort_values('Count')
    data['Cumulative Percentage'] = data['Count'].cumsum() / total_occurrences * 100

    # Creating the cumulative distribution plot
    plt.figure(figsize=(10, 6))
    sns.lineplot(x='Song', y='Cumulative Percentage', data=data)
    plt.xticks(rotation=90)
    plt.title('Cumulative Distribution of Song Occurrences INCLUDING UNKNOWN')
    plt.xlabel('Songs')
    plt.ylabel('Cumulative Percentage')
    plt.tight_layout()
    plt.savefig(f'{str(fp)}/cdf_song_occurrences_INCLUDING_UNKNOWN.pdf', bbox_inches='tight')
    plt.close()

    # Extract data from the Counter object
    songs, counts = zip(*song_occurences_in_collective.items())
    total_occurrences = sum(counts)

    # Create a DataFrame
    data = pd.DataFrame({'Song': songs, 'Count': counts}).sort_values('Count')
    data['Cumulative Percentage'] = data['Count'].cumsum() / total_occurrences * 100

    # Creating the cumulative distribution plot
    plt.figure(figsize=(10, 6))
    sns.lineplot(x='Song', y='Cumulative Percentage', data=data)
    plt.xticks(rotation=90)
    plt.title('Cumulative Distribution of Song Occurrences')
    plt.xlabel('Songs')
    plt.ylabel('Cumulative Percentage')
    plt.tight_layout()
    plt.savefig(f'{str(fp)}/cdf_song_occurrences_collective.pdf', bbox_inches='tight')
    plt.close()


    collective_action_description = f"the collective consists of {budget} playlists ## with AVG={np.mean([len(c) for c in collective])} / MIN={np.min([len(c) for c in collective])} / MAX={np.max([len(c) for c in collective])} / MEDIAN={np.median([len(c) for c in collective])} playlists lengths ## 3 most_common song_occurences_in_train: {song_occurences_in_train.most_common(3)} ## 3 most_common song_occurences_in_collective: {song_occurences_in_collective.most_common(3)}"

    # Initialize a defaultdict with default value of 0 for each key
    targeted_seed_songs = defaultdict(lambda: 0)
    target_x = get_targetx(signal_planting_strategy)
    percentile = get_percentile(signal_planting_strategy)

    if "optimal" not in signal_planting_strategy:
        #pdb.set_trace()
        if "atthebeginning" in signal_planting_strategy:
            # always plant at the beginning of the playlist
            for playlist in collective:
                duplicate_playlists_to_insert = insert_songs_in_playlist(playlist, 0, get_signals(playlist[0]), rng, insertbefore, replaceseed, duplicateseed, duplicate_playlists_to_insert)
                targeted_seed_songs[playlist[0]] += 1
                signals_planted += 1
            collective_action_description += f" ####### added target song {signals_planted} times at the BEGINNING of every playlist controlled by the collective"
        elif "atindex1" in signal_planting_strategy:
            # always plant at index 1
            for playlist in collective:
                index_to_insert = min(1, len(playlist))
                duplicate_playlists_to_insert = insert_songs_in_playlist(playlist, index_to_insert, get_signals(playlist[index_to_insert-1]), rng, insertbefore, replaceseed, duplicateseed, duplicate_playlists_to_insert)
                targeted_seed_songs[playlist[index_to_insert-1]] += 1
                signals_planted += 1
            collective_action_description += f" ####### added target song {signals_planted} times at INDEX 1 of every playlist controlled by the collective"
        elif "atindex3" in signal_planting_strategy:
            # always plant at index 3
            for playlist in collective:
                index_to_insert = min(3, len(playlist))
                duplicate_playlists_to_insert = insert_songs_in_playlist(playlist, index_to_insert, get_signals(playlist[index_to_insert-1]), rng, insertbefore, replaceseed, duplicateseed, duplicate_playlists_to_insert)
                targeted_seed_songs[playlist[index_to_insert-1]] += 1
                signals_planted += 1
            collective_action_description += f" ####### added target song {signals_planted} times at INDEX 3 of every playlist controlled by the collective"
        elif "atindex5" in signal_planting_strategy:
            # always plant at index 5
            for playlist in collective:
                index_to_insert = min(5, len(playlist))
                duplicate_playlists_to_insert = insert_songs_in_playlist(playlist, index_to_insert, get_signals(playlist[index_to_insert-1]), rng, insertbefore, replaceseed, duplicateseed, duplicate_playlists_to_insert)
                targeted_seed_songs[playlist[index_to_insert-1]] += 1
                signals_planted += 1
            collective_action_description += f" ####### added target song {signals_planted} times at INDEX 5 of every playlist controlled by the collective"
        elif "atindex7" in signal_planting_strategy:
            # always plant at index 7
            for playlist in collective:
                index_to_insert = min(7, len(playlist))
                duplicate_playlists_to_insert = insert_songs_in_playlist(playlist, index_to_insert, get_signals(playlist[index_to_insert-1]), rng, insertbefore, replaceseed, duplicateseed, duplicate_playlists_to_insert)
                targeted_seed_songs[playlist[index_to_insert-1]] += 1
                signals_planted += 1
            collective_action_description += f" ####### added target song {signals_planted} times at INDEX 7 of every playlist controlled by the collective"
        elif "attheend" in signal_planting_strategy:
            # always plant at the end of the playlist
            for playlist in collective:
                targeted_seed_songs[playlist[-1]] += 1
                duplicate_playlists_to_insert = insert_songs_in_playlist(playlist, len(playlist), get_signals(playlist[-1]), rng, insertbefore, replaceseed, duplicateseed, duplicate_playlists_to_insert)
                signals_planted += 1
            collective_action_description += f" ####### added target song {signals_planted} times at the END of every playlist controlled by the collective"
        elif "atindex1" in signal_planting_strategy:
            # always plant at index 1 of the playlist
            for playlist in collective:
                duplicate_playlists_to_insert = insert_songs_in_playlist(playlist, 1, get_signals(playlist[0]), rng, insertbefore, replaceseed, duplicateseed, duplicate_playlists_to_insert)
                targeted_seed_songs[playlist[0]] += 1
                signals_planted += 1
            collective_action_description += f" ####### added target song {signals_planted} times at INDEX 1 of every playlist controlled by the collective"
        elif "dirlof2x" in signal_planting_strategy or "dirlof3x" in signal_planting_strategy:
            if "dirlof2x" in signal_planting_strategy:
                x_times = 2
            elif "dirlof3x" in signal_planting_strategy:
                x_times = 3
            # get all songs that occur at least x times in the collective
            songs_occurring_x_times_in_collective = [s for s, count in song_occurences_in_collective.items() if count <= x_times]
            # sort by number of occurences in the collective (decreasing) and then by number of occurrences in the training data (increasing)
            songs_occurring_x_times_in_collective = sorted(songs_occurring_x_times_in_collective, key=lambda x: (-song_occurences_in_collective[x], song_occurences_in_train[x]))
            available_playlist_indices = list(range(len(collective)))
            for song in songs_occurring_x_times_in_collective:
                count = song_occurences_in_collective[song]
                # if there are count playlists in the available_playlist_indices, plant the signal in all of them. otherwise, continue.
                playlists_with_song = [i for i in available_playlist_indices if song in collective[i]]
                if len(playlists_with_song) >= count:
                    for playlist_index in playlists_with_song[:count]:
                        # Plant song
                        song_index = collective[playlist_index].index(song)
                        duplicate_playlists_to_insert = insert_songs_in_playlist(collective[playlist_index], song_index + 1, get_signals(song), rng, insertbefore, replaceseed, duplicateseed, duplicate_playlists_to_insert)
                        targeted_seed_songs[song] += 1
                        signals_planted += 1
                        # Remove the index from the list of available indices
                        available_playlist_indices.remove(playlist_index)
            # if there are available_playlist_indices left, insert the signal at a random position in all of them
            if len(available_playlist_indices) > 0:
                for i in available_playlist_indices:
                    # Plant song at random position
                    index = rng.integers(len(collective[i]))
                    song = collective[i][index]
                    duplicate_playlists_to_insert = insert_songs_in_playlist(collective[i], index + 1, get_signals(song), rng, insertbefore, replaceseed, duplicateseed, duplicate_playlists_to_insert)
                    targeted_seed_songs[song] += 1
                    signals_planted += 1
        elif "dirlof" in signal_planting_strategy:
            scraped_streams = None
            artist_popularities = None
            # always plant after least famous song in the playlist
            for playlist in collective:
                if "0trainfraction_artistpopularities" in signal_planting_strategy:
                    artist_popularities, least_famous_song = get_song_with_lowest_artist_popularity(playlist, all_song_info, artist_popularities, rng)
                elif "0trainfraction_scrapedstreams" in signal_planting_strategy:
                    scraped_streams, least_famous_song = get_song_with_fewest_streams(playlist, all_song_info, scraped_streams, rng)
                elif "dirlof_0trainfraction_randomonce" in signal_planting_strategy:
                    least_famous_song = playlist[rng.integers(len(playlist))]
                elif "dirlof_0trainfraction_indirect" in signal_planting_strategy:
                    # TODO: fix, we want to check indirect anchor candidates that are preceeded by an infrequent song!
                    insertbefore = True
                    target_x = 1
                    candidates_in_playlist = []
                    while len(candidates_in_playlist == 1):
                        candidates_in_playlist = [s for s in playlist if song_occurences_in_collective[s] == target_x]
                        target_x += 1
                    least_famous_song = select_one_of_most_freq_songs(candidates_in_playlist, song_occurences_in_collective, 1, rng)
                elif target_x:
                    songs_occurring_x_times_in_collective = []
                    my_target_x = target_x
                    while len(songs_occurring_x_times_in_collective) == 0:
                        # get subset of songs in playlist that occur exactly x times in the collective and that have not yet been targeted x times
                        songs_occurring_x_times_in_collective = [s for s in playlist if (song_occurences_in_collective[s] == my_target_x)]
                        #print(f"target_x is {target_x}, my_target_x is {my_target_x}, nr of songs_occurring_x_times_in_collective is {len(songs_occurring_x_times_in_collective)}")
                        my_target_x -= 1
                        if my_target_x < 0:
                            break
                    if len(songs_occurring_x_times_in_collective) == 0:
                        # no songs in playlist occur exactly x times in the collective, so select the least famous song in the playlist
                        least_famous_song = min(playlist, key=lambda x: song_occurences_in_train[x])
                    else:
                        # select the least famous song among those occurring at least x times in the collective
                        least_famous_song = min(songs_occurring_x_times_in_collective, key=lambda x: song_occurences_in_train[x])
                else:
                    least_famous_song = min(playlist, key=lambda x: song_occurences_in_train[x])
                index = playlist.index(least_famous_song)
                duplicate_playlists_to_insert = insert_songs_in_playlist(playlist, index + 1, get_signals(least_famous_song), rng, insertbefore, replaceseed, duplicateseed, duplicate_playlists_to_insert)
                targeted_seed_songs[least_famous_song] += 1
                signals_planted += 1
                print(f"planting target song before/after index {index + 1} in playlist, targeting the LEAST FAMOUS SONG: {least_famous_song}")
            collective_action_description += f" ####### added target song {signals_planted} times before/after LEAST FAMOUS SONG in every playlist controlled by the collective that OCCUR EXACTLY {target_x} times in the collective (or less if impossible)"

        elif "randominfirst10random" in signal_planting_strategy:
            # every member of the collective randomly places the target song in the playlist
            for playlist in collective:
                # get a random index in the first 10 songs of the playlist
                random_index = rng.integers(min(10, len(playlist)))
                targeted_seed_songs[playlist[random_index]] += 1
                #pdb.set_trace()
                duplicate_playlists_to_insert = insert_songs_in_playlist(playlist, random_index + 1, get_signals(playlist[random_index]), rng, insertbefore, replaceseed, duplicateseed, duplicate_playlists_to_insert)
                signals_planted += 1
            collective_action_description += f" ####### added {signals_planted} at RANDOM positions in THE FIRST 10 POSITIONS OF every playlist controlled by the collective"
        elif "randominlast10random" in signal_planting_strategy:
            # every member of the collective randomly places the target song in the playlist
            for playlist in collective:
                # get a random index in the last 10 songs of the playlist
                random_index = rng.integers(max(0, len(playlist)-10), len(playlist))
                targeted_seed_songs[playlist[random_index]] += 1
                #pdb.set_trace()
                duplicate_playlists_to_insert = insert_songs_in_playlist(playlist, random_index + 1, get_signals(playlist[random_index]), rng, insertbefore, replaceseed, duplicateseed, duplicate_playlists_to_insert)
                signals_planted += 1
            collective_action_description += f" ####### added {signals_planted} at RANDOM positions in THE LAST 10 POSITIONS OF every playlist controlled by the collective"
        elif "random" in signal_planting_strategy:
            # every member of the collective randomly places the target song in the playlist
            for playlist in collective:
                random_index = rng.integers(len(playlist))
                targeted_seed_songs[playlist[random_index]] += 1
                #pdb.set_trace()
                duplicate_playlists_to_insert = insert_songs_in_playlist(playlist, random_index + 1, get_signals(playlist[random_index]), rng, insertbefore, replaceseed, duplicateseed, duplicate_playlists_to_insert)
                signals_planted += 1
            collective_action_description += f" ####### added {signals_planted} at RANDOM positions in every playlist controlled by the collective"
        elif "inclust" in signal_planting_strategy:
            # every member of the collective places the target song after the most famous song in the playlist (w.r.t. the number of occurences in the entire collective)
            for playlist in collective:
                most_listened_to_track = select_one_of_most_freq_songs(playlist, song_occurences_in_collective, 1, rng)
                most_listened_to_track_index = playlist.index(most_listened_to_track)
                targeted_seed_songs[most_listened_to_track] += 1
                duplicate_playlists_to_insert = insert_songs_in_playlist(playlist, most_listened_to_track_index + 1, get_signals(most_listened_to_track), rng, insertbefore, replaceseed, duplicateseed, duplicate_playlists_to_insert)
                signals_planted += 1
            collective_action_description += f" ####### added {signals_planted} in total. looped through all playlists controlled by the collective and added the signal after the most listened to song by the collective (the one with the most occurences in the entire collective) that is present in the playlist. this strategy maximizes the number of signals planted after the same song"
        
    print("collective_action_description:", collective_action_description)



    if len(dict(targeted_seed_songs).keys()) >0:
        save_targeted_seed_songs_as_csv(targeted_seed_songs, fp, all_song_info)
        
    save_modified_playlists(collective, signals, fp, all_song_info, signal_planting_strategy)


    # plot candidates in collective and target songs
    # Convert counters to DataFrames
    df_total = pd.DataFrame.from_dict(song_occurences_in_train, orient='index', columns=['Occurrences'])
    df_collective = pd.DataFrame.from_dict(song_occurences_in_collective, orient='index', columns=['Occurrences in Collective'])

    # Calculate ranks in the total distribution
    df_total['Rank'] = df_total['Occurrences'].rank(pct=False)

    # Merge data on song index
    merged_data = df_collective.join(df_total['Rank'], how='inner')


    # Convert targeted_seed_songs to DataFrame
    df_targeted = pd.DataFrame.from_dict(targeted_seed_songs, orient='index', columns=['Targeted Count'])

    # Merge targeted data with the existing merged_data
    final_data = merged_data.join(df_targeted, how='left')
    final_data['Targeted Count'].fillna(0, inplace=True)

    # Creating the scatter plot
    plt.figure(figsize=(8, 6))

    # Calculate the maximum value for 'Occurrences in Collective' and set a buffer
    max_occurrences = final_data['Occurrences in Collective'].max()
    buffer = max_occurrences * 0.05  # 5% buffer to ensure annotations fit within the plot

    # Set the y-axis limits
    plt.ylim(0, max_occurrences + buffer)
    
    # Plot for non-targeted songs
    sns.scatterplot(data=final_data[final_data['Targeted Count'] == 0], x='Rank', y='Occurrences in Collective', color='blue', label='Non-Targeted', alpha=0.2)

    # Annotate for targeted songs
    for _, row in final_data[final_data['Targeted Count'] > 0].iterrows():
        plt.text(row['Rank'], row['Occurrences in Collective'], int(row['Targeted Count']), color='red', verticalalignment='bottom', horizontalalignment='right', alpha=0.3)

    plt.title('Song Occurrences in Collective vs. Rank (Targeted Songs Marked)')
    plt.xlabel('Rank in Total Distribution')
    plt.ylabel('Occurrences in Collective')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{str(fp)}/rank_and_occurence_of_songs_in_collective_with_targeted_marked.pdf', bbox_inches='tight')
    plt.close()



    ####
    ## plot expected occurrence instead of percentiles
    ####

    # Convert counters to DataFrames
    df_total = pd.DataFrame.from_dict(song_occurences_in_train, orient='index', columns=['Occurrences'])
    df_collective = pd.DataFrame.from_dict(song_occurences_in_collective, orient='index', columns=['Occurrences in Collective'])

    # multiply occurrences with alpha to get expected occurrences in collective
    alpha = budget / (len(known_training_data) + len(unknown_training_data))
    print("alpha:", alpha)
    print("len(known_training_data):", len(known_training_data))
    print("len(unknown_training_data):", len(unknown_training_data))
    df_total[r'$\alpha$ occurrences'] = df_total['Occurrences'] * alpha

    # Merge data on song index
    merged_data = df_collective.join(df_total[r'$\alpha$ occurrences'], how='inner')


    # Convert targeted_seed_songs to DataFrame
    df_targeted = pd.DataFrame.from_dict(targeted_seed_songs, orient='index', columns=['Targeted Count'])

    # Merge targeted data with the existing merged_data
    final_data = merged_data.join(df_targeted, how='left')
    final_data['Targeted Count'].fillna(0, inplace=True)

    # Creating the scatter plot
    plt.figure(figsize=(8, 6))

    # Calculate the maximum value for 'Occurrences in Collective' and set a buffer
    max_occurrences = final_data['Occurrences in Collective'].max()
    buffer = max_occurrences * 0.05  # 5% buffer to ensure annotations fit within the plot

    # Set the y-axis limits
    plt.ylim(0, max_occurrences + buffer)
    
    # Plot for non-targeted songs
    sns.scatterplot(data=final_data[final_data['Targeted Count'] == 0], x=r'$\alpha$ occurrences', y='Occurrences in Collective', color='blue', label='Non-Targeted', alpha=0.2)

    # Annotate for targeted songs
    for _, row in final_data[final_data['Targeted Count'] > 0].iterrows():
        plt.text(row[r'$\alpha$ occurrences'], row['Occurrences in Collective'], int(row['Targeted Count']), color='red', verticalalignment='bottom', horizontalalignment='right', alpha=0.3)

    plt.title('Song Occurrences in Collective vs. overall (Targeted Songs Marked)')
    plt.xlabel(r'$\alpha$ ccurrences in Total Distribution')
    plt.ylabel('Occurrences in Collective')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{str(fp)}/occurence_of_songs_in_collective_and_overall_with_targeted_marked.pdf', bbox_inches='tight')
    plt.close()


    ####
    ## plot expected occurrence INCLUDING UNKNOWN
    ####

    if len(unknown_training_data) > 0:

        # Convert counters to DataFrames
        df_total = pd.DataFrame.from_dict(song_occurences_in_train_INCLUDING_UNKNOWN, orient='index', columns=['Occurrences'])
        df_collective = pd.DataFrame.from_dict(song_occurences_in_collective, orient='index', columns=['Occurrences in Collective'])

        # multiply occurrences with alpha to get expected occurrences in collective
        alpha = budget / (len(known_training_data) + len(unknown_training_data))
        print("alpha:", alpha)
        print("len(known_training_data):", len(known_training_data))
        print("len(unknown_training_data):", len(unknown_training_data))
        df_total[r'$\alpha$ occurrences'] = df_total['Occurrences'] * alpha

        # Merge data on song index
        merged_data = df_collective.join(df_total[r'$\alpha$ occurrences'], how='inner')


        # Convert targeted_seed_songs to DataFrame
        df_targeted = pd.DataFrame.from_dict(targeted_seed_songs, orient='index', columns=['Targeted Count'])

        # Merge targeted data with the existing merged_data
        final_data = merged_data.join(df_targeted, how='left')
        final_data['Targeted Count'].fillna(0, inplace=True)

        # Creating the scatter plot
        plt.figure(figsize=(8, 6))

        # Calculate the maximum value for 'Occurrences in Collective' and set a buffer
        max_occurrences = final_data['Occurrences in Collective'].max()
        buffer = max_occurrences * 0.05  # 5% buffer to ensure annotations fit within the plot

        # Set the y-axis limits
        plt.ylim(0, max_occurrences + buffer)
        
        # Plot for non-targeted songs
        sns.scatterplot(data=final_data[final_data['Targeted Count'] == 0], x=r'$\alpha$ occurrences', y='Occurrences in Collective', color='blue', label='Non-Targeted', alpha=0.2)

        # Annotate for targeted songs
        for _, row in final_data[final_data['Targeted Count'] > 0].iterrows():
            plt.text(row[r'$\alpha$ occurrences'], row['Occurrences in Collective'], int(row['Targeted Count']), color='red', verticalalignment='bottom', horizontalalignment='right', alpha=0.3)

        plt.title('Song Occurrences in Collective vs. overall (Targeted Songs Marked)')
        plt.xlabel(r'$\alpha$ ccurrences in Total Distribution')
        plt.ylabel('Occurrences in Collective')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'{str(fp)}/occurence_of_songs_in_collective_and_overall_with_targeted_marked_INCLUDING_UNKNOWN.pdf', bbox_inches='tight')
        plt.close()


    print(f"Total GB of RAM used: {psutil.virtual_memory().used / (1024**3)}")
    
    return collective, duplicate_playlists_to_insert, signals_planted, collective_action_description



def save_modified_playlists(collective, signals, fp, all_song_info, signal_planting_strategy):
    # save targeted seed song contexts to disk
    signal_context = []
    playlist_continuations = []
    poisoned_playlists = []
    
    signal = signals[0]
    print(f"I look at the signal: {signal}")
    
    if "poisonblbase" in signal_planting_strategy:
        csv_file_path = Path(fp, f"poisoned_playlists.csv")
        all_song_info[signal] = {"track_uri": "TARGET_SONG"}
        for index, playlist in enumerate(collective):
            poisoned_playlists.append([all_song_info[song_id]["track_uri"] for song_id in playlist])
        print(f"Writing results to {csv_file_path}")
        # Append the new line with results to the csv file
        with open(csv_file_path, 'w', newline='') as file:
            writer = csv.writer(file)
            for poisoned_p in poisoned_playlists:
                writer.writerow(poisoned_p)
    else:
        csv_file_path = Path(fp, f"modified_playlists_signal_context.csv")
        csv_file_path_playlist_continuations = Path(fp, f"modified_playlists_continuation.csv")
        for index, playlist in enumerate(collective):
            if signal in playlist:
                # get everything before the signal
                signal_context.append([all_song_info[song_id]["track_uri"] for song_id in playlist[:playlist.index(signal)]])
                # get everything after (and including) the signal
                playlist_continuations.append([all_song_info[song_id]["track_uri"] for song_id in playlist[playlist.index(signal)+1:]])

        print(f"Writing results to {csv_file_path}")
        # Append the new line with results to the csv file
        with open(csv_file_path, 'w', newline='') as file:
            writer = csv.writer(file)
            for playlist_context in signal_context:
                writer.writerow(playlist_context)

        print(f"Writing results to {csv_file_path_playlist_continuations}")
        # Append the new line with results to the csv file
        with open(csv_file_path_playlist_continuations, 'w', newline='') as file:
            writer = csv.writer(file)
            for continuation in playlist_continuations:
                writer.writerow(continuation)



def save_targeted_seed_songs_as_csv(targeted_seed_songs, fp, all_song_info):
    # save new_playlist_to_song to disk
    csv_file_path = Path(fp, f"targeted_seed_songs.csv")
    headers = ["nr_of_times_this_seed_was_targeted", "seed_song_uri", "seed_song_name", "artist_uri", "artist_name"]
    if not os.path.isfile(csv_file_path):
        print(f"Writing result headers to {csv_file_path}")
        with open(csv_file_path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(headers)

    print(f"Writing results to {csv_file_path}")
    try:
        # Append the new line with results to the csv file
        with open(csv_file_path, 'a', newline='') as file:
            writer = csv.writer(file)
            # sort targeted_seed_songs by values
            targeted_seed_songs = {k: v for k, v in sorted(targeted_seed_songs.items(), key=lambda item: item[1], reverse=True)}
            # Looping through the dictionary, sorted by values
            for seed_song_id, nr in sorted(targeted_seed_songs.items(), key=lambda item: item[1], reverse=True):
                row = [nr, all_song_info[seed_song_id]["track_uri"], all_song_info[seed_song_id]["track_name"], all_song_info[seed_song_id]["artist_uri"], all_song_info[seed_song_id]["artist_name"]]
                writer.writerow(row)
    except:
        print("Error while writing to csv file.")



@utils.timer
def save_data_with_signal(collective_with_signal, collective_playlistpids, all_data, original_dataset_filepath, new_path, signal_track_ids, signal_metadata, all_song_info, track_id_by_track_uri, rng, duplicate_playlists_to_insert=[]):
    # Create a dictionary for easy access to playlists by their pid
    playlists_dict = {playlist['pid']: playlist for playlist in all_data}
    

    for signal_track_id in signal_track_ids:
        # add the track info of the signal track to the all_song_info dict
        # get key from track_id_by_track_uri where value is signal_track_id
        signal_track_uri = [k for k, v in track_id_by_track_uri.items() if v == signal_track_id][0]
        new_track = {
            "artist_name": signal_metadata["artist_name"],
            "track_uri": signal_track_uri,
            "artist_uri": signal_metadata["artist_uri"],
            "track_name": signal_metadata["track_name"],
            "album_uri": signal_metadata["album_uri"],
            "duration_ms": signal_metadata["duration_ms"],
            "album_name": signal_metadata["album_name"]
        }
        all_song_info[signal_track_id] = new_track

    #pdb.set_trace()
    # Iterate over each playlist in collective_with_signal
    for playlist_signal, pid in zip(collective_with_signal, collective_playlistpids):
        # Get the corresponding playlist
        current_playlist = playlists_dict[pid]

        # Initialize a set to keep track of unique albums and artists
        albums = set()
        artists = set()
        duration = 0

        # Initialize a list to store the updated tracks and a counter for the original tracks
        updated_tracks = []

        # Iterate over each track_id in the playlist_signal
        for i, track_id in enumerate(playlist_signal):
            # this song was added as part of the double placement strategy, thus, grab metadata from the original dataset
            track = all_song_info[track_id]
            track['pos'] = i
            updated_tracks.append(track)

            # Update the set of unique albums and artists and the total duration
            albums.add(updated_tracks[-1]['album_uri'])
            artists.add(updated_tracks[-1]['artist_uri'])
            duration += updated_tracks[-1]['duration_ms']

        # Update the tracks in the playlist
        current_playlist['tracks'] = updated_tracks

        # Update the playlist-level information
        current_playlist['num_tracks'] = max(len(updated_tracks), current_playlist['num_tracks'])
        current_playlist['num_albums'] = max(len(albums), current_playlist['num_albums'])
        current_playlist['duration_ms'] = max(duration, current_playlist['duration_ms'])
        current_playlist['num_artists'] = max(len(artists), current_playlist['num_artists'])
    
    random_playlist_pids = []
    while len(duplicate_playlists_to_insert) > len(random_playlist_pids):
        # TODO: make sure this does not run forever
        # get a random playlist from all_data, using rng
        random_playlist = rng.choice(all_data)
        random_playlist_pid = random_playlist["pid"]
        if random_playlist_pid not in collective_playlistpids:
            random_playlist_pids.append(random_playlist_pid)

        
    #pdb.set_trace()
    # Iterate over each playlist in duplicate_playlists_to_insert
    for playlist_signal, pid in zip(duplicate_playlists_to_insert, random_playlist_pids):
        # TODO: make function as this is basically the same as above
        # Get the corresponding playlist
        current_playlist = playlists_dict[pid]

        # Initialize a set to keep track of unique albums and artists
        albums = set()
        artists = set()
        duration = 0

        # Initialize a list to store the updated tracks and a counter for the original tracks
        updated_tracks = []

        # Iterate over each track_id in the playlist_signal
        for i, track_id in enumerate(playlist_signal):
            # this song was added as part of the double placement strategy, thus, grab metadata from the original dataset
            track = all_song_info[track_id]
            track['pos'] = i
            updated_tracks.append(track)

            # Update the set of unique albums and artists and the total duration
            albums.add(updated_tracks[-1]['album_uri'])
            artists.add(updated_tracks[-1]['artist_uri'])
            duration += updated_tracks[-1]['duration_ms']

        # Update the tracks in the playlist
        current_playlist['tracks'] = updated_tracks

        # Update the playlist-level information
        current_playlist['num_tracks'] = max(len(updated_tracks), current_playlist['num_tracks'])
        current_playlist['num_albums'] = max(len(albums), current_playlist['num_albums'])
        current_playlist['duration_ms'] = max(duration, current_playlist['duration_ms'])
        current_playlist['num_artists'] = max(len(artists), current_playlist['num_artists'])

        print(f" added duplicate playlist with pid {pid} to the dataset // len(duplicate_playlists_to_insert): {len(duplicate_playlists_to_insert)}, len(random_playlist_pids): {len(random_playlist_pids)}")
    print(f"collective_playlistpids: {collective_playlistpids}")
    print(f"random_playlist_pids: {random_playlist_pids}")
    # check if collective_playlistpids and random_playlist_pids have any overlap
    overlap_playlist_pids = set(collective_playlistpids) & set(random_playlist_pids)
    assert not overlap_playlist_pids, f"Overlap in playlist pids for collective_playlistpids and random_playlist_pids: {overlap_playlist_pids}"

    
    return [playlists_dict[playlist['pid']] for playlist in all_data]
