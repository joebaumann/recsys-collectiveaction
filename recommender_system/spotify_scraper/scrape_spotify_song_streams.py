import pandas as pd
import os
import tqdm
import json
from collections import defaultdict, Counter
import re
import requests
from bs4 import BeautifulSoup
from difflib import SequenceMatcher
import base64
import requests
import datetime
from python_api_credentials import CLIENT_ID, CLIENT_SECRET
import pdb
import seaborn as sns
import matplotlib.pyplot as plt
from multiprocessing import Pool
from itertools import islice



def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()


class SpotifyAPI(object):
    """docstring for SpotifyAPI"""
    def __init__(self, client_id, client_secret):
        self.client_id = client_id
        self.client_secret = client_secret
        self.access_token = None
        self.token_type = None
        self.token_expires = datetime.datetime.now()
        self.request_header = None

    
    def get_client_credentials(self):
        credentials = f"{self.client_id}:{self.client_secret}"
        encoded_credentials = base64.b64encode(credentials.encode())
        print("encoded_credentials:", encoded_credentials)
        return encoded_credentials.decode()
    
    def authenticate(self):
        encoded_credentials = self.get_client_credentials()
        token_url = "https://accounts.spotify.com/api/token"
        token_data = {
            "grant_type": "client_credentials"
        }
        token_headers = {
            "Authorization": f"Basic {encoded_credentials}"
        }
        print("token_headers:", token_headers)
        r = requests.post(token_url, data=token_data, headers=token_headers)
        token_response = r.json()
        print("token_response:", token_response)
        self.access_token = token_response["access_token"]
        self.token_type = token_response["token_type"]
        expires_in = token_response["expires_in"]
        print("expires_in:", expires_in)
        self.token_expires = datetime.datetime.now() + datetime.timedelta(seconds=expires_in)
        print("expires:", self.token_expires.strftime("%m/%d/%Y, %H:%M:%S"))
        print("token_response:", token_response)
        self.request_header = {
            "Authorization": f"{self.token_type} {self.access_token}"
        }
        return token_response

    def make_api_request(self, endpoint, data=None, params=None):
        if self.token_expires <= datetime.datetime.now():
            self.authenticate()
        url = f"https://api.spotify.com/v1/{endpoint}"
        r = requests.get(url, data=data, headers=self.request_header, params=params)
        #print("url:", url, "r:", r)
        return r.json()



def process_batch(batch, batch_id):
    filepath = f"spotify_scraper/scraped_streams_batch_{batch_id}.csv"
    
    print(f"Processing batch {batch_id}")
    with open(filepath, "a") as f:
        for (song_uri, track_name) in tqdm.tqdm(batch):
            url, stream_count = get_spotify_streams(song_uri, track_name)
            f.write(f"{song_uri},{stream_count}\n")


def chunks(data, size):
    it = iter(data)
    for i in range(0, len(data), size):
        yield list(islice(it, size))

# def chunks(data, size):
#     it = iter(data)
#     for i in range(0, len(data), size):
#         yield {k: data[k] for k in islice(it, size)}
    
    


def get_spotify_streams(song_uri, track_name):
    url = f"https://open.spotify.com/track/{song_uri}"
    response = requests.get(url)
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



def get_all_playlists(raw_path, nr_of_data_files=-1):

    print("processing MPD to organize collective and to define train, eval, and test set indices")
    #all_playlists = []
    track_id = 0
    all_playlists_only_track_ids = []
    all_song_info = {}
    count = 0

    track_id_by_track_uri = {}

    filenames = os.listdir(raw_path)
    for filename in tqdm.tqdm(sorted(filenames, key=str)):
        if filename.startswith("mpd.slice.") and filename.endswith(".json"):
            fullpath = os.sep.join((raw_path, filename))
            f = open(fullpath)
            js = f.read()
            f.close()
            mpd_slice = json.loads(js)

            playlists_one_file = mpd_slice["playlists"]

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

            #all_playlists.extend(playlists_one_file)
            count += 1
             
            if nr_of_data_files != -1 and count > nr_of_data_files:
                break
    
    # save all_song_info to disk
    with open(f"spotify_scraper/all_song_info.json", "w") as f:
        json.dump(all_song_info, f, indent=4)
    
    return all_playlists_only_track_ids, all_song_info
    #return Counter([track_id for playlist in all_playlists_only_track_ids for track_id in playlist]), all_song_info

def get_song_frequencies(percentages_of_data, raw_path, nr_of_data_files):
    # try to load song_frequencies and all_song_info from disk. otherwise process the MPD

    could_load_all_song_freqs_from_disk = True
    song_frequencies = {}
    # # TODO: check if song_frequencies already exist and load them from disk if so
    for perc in percentages_of_data:
        if os.path.exists(f"spotify_scraper/song_frequencies_{perc}_of_data.json"):
            with open(f"spotify_scraper/song_frequencies_{perc}_of_data.json", "r") as f:
                loaded_freqs = json.load(f)
                loaded_freqs = {int(song_id): freq for song_id, freq in loaded_freqs.items()}
                song_frequencies[perc] = loaded_freqs
            print(f"Loaded song_frequencies_{perc}_of_data.json from disk")
        else:
            could_load_all_song_freqs_from_disk = False
            break
    
    if os.path.exists(f"spotify_scraper/all_song_info.json"):
        with open(f"spotify_scraper/all_song_info.json", "r") as f:
            # load all_song_info from disk, key is track_id as int, value is track_info
            all_song_info = json.load(f)
            # convert track_id from string to int
            all_song_info = {int(track_id): track_info for track_id, track_info in all_song_info.items()}
        print(f"Loaded all_song_info.json from disk")
    else:
        could_load_all_song_freqs_from_disk = False
            
    if not could_load_all_song_freqs_from_disk:
        # not all song_freqs could be loaded from disk, so we need to process the MPD
        all_playlists, all_song_info = get_all_playlists(raw_path, nr_of_data_files)
        song_frequencies = process_data_to_get_song_frequencies(all_playlists, percentages_of_data)

    print(f"Total number of songs: {len(song_frequencies[1])}")

    
    return song_frequencies, all_song_info

def process_data_to_get_song_frequencies(all_playlists, percentages_of_data):
    song_frequencies = {}
    for perc in percentages_of_data:
        song_frequencies[perc] = defaultdict(lambda: 0)

    for i, playlist in enumerate(all_playlists):
        for track in playlist:
            for perc in percentages_of_data:
                if i / len(all_playlists) <= perc:
                    song_frequencies[perc][track] += 1
    
    # save song_frequencies to disk
    for perc in song_frequencies:
        with open(f"spotify_scraper/song_frequencies_{perc}_of_data.json", "w") as f:
            json.dump(song_frequencies[perc], f, indent=4)
    
    return song_frequencies


def scrape_streams(song_frequencies, all_song_info):

    filepath_aggregated = "spotify_scraper/scraped_streams_aggregated.csv"

    if os.path.exists(filepath_aggregated):
        with open(filepath_aggregated, "r") as f:
            # load csv file into pd dataframe
            scraped_streams = pd.read_csv(f)
        print(f"Loaded scraped_streams.csv from disk")
        filepath_aggregated = "spotify_scraper/scraped_streams_aggregated_NEW.csv"
        #return scraped_streams
    
    # print(f"Number of songs in song_frequencies: {len(song_frequencies)}")


    # # exlude all songs in song_frequencies that are already in scraped_streams
    # filtered_song_frequencies = {}
    # for song, freq in tqdm.tqdm(song_frequencies.items()):
    #     track_uri = all_song_info[int(song)]["track_uri"].replace("spotify:track:", "")
    #     if track_uri not in scraped_streams["track_uri"].values:
    #         filtered_song_frequencies[song] = freq
    #     else:
    #         pass
    #         #print(f"Song already in scraped streams: {track_uri}")

    #     if len(filtered_song_frequencies) >= 1000:
    #         break

    # # Now filtered_song_frequencies contains only the songs not in scraped_streams

    # print(f"Number of already scraped songs: {len(song_frequencies) - len(filtered_song_frequencies)}")
    # print(f"Number of songs to scrape: {len(filtered_song_frequencies)}")

    num_processes = 30

    # Splitting song_frequencies into batches
    song_batches = list(chunks([(all_song_info[int(song)]["track_uri"].replace("spotify:track:", ""), all_song_info[int(song)]["track_name"]) for song in song_frequencies.keys()], len(song_frequencies) // num_processes + 1))
    #song_batches = list(chunks([(all_song_info[int(song)]["track_uri"].replace("spotify:track:", ""), all_song_info[int(song)]["track_name"]) for song in filtered_song_frequencies.keys()], len(filtered_song_frequencies) // num_processes + 1))

    # Initialize files for each batch
    for i in range(num_processes):
        filepath = f"spotify_scraper/scraped_streams_batch_{i}.csv"
        if not os.path.exists(filepath):
            with open(filepath, "w") as f:
                f.write("track_uri,streams\n")

    # Multiprocessing
    print("Starting multiprocessing")
    with Pool(processes=num_processes) as pool:
        for i, batch in enumerate(song_batches):
            pool.apply_async(process_batch, args=(batch, i))

        pool.close()
        pool.join()
    
    print("Finished multiprocessing")

    # Aggregating results from all files
    aggregated_data = pd.DataFrame()
    for i in range(num_processes):
        filepath = f"spotify_scraper/scraped_streams_batch_{i}.csv"
        batch_data = pd.read_csv(filepath)
        aggregated_data = pd.concat([aggregated_data, batch_data])

    # Save the aggregated data to a single file
    aggregated_data.to_csv(filepath_aggregated, index=False)

    return aggregated_data


def get_artist_popularities(song_frequencies, all_song_info, artist_batch_size=50):

    filepath = f"spotify_scraper/artist_popularities.json"

    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            artist_popularities = json.load(f)
        print(f"Loaded artist_popularities.json from disk")
        return artist_popularities
    
    pdb.set_trace()

    spotify_client = SpotifyAPI(CLIENT_ID, CLIENT_SECRET)
    spotify_client.authenticate()

    artist_popularities = defaultdict(lambda: 0)
    artist_batch = []
    for song in tqdm.tqdm(song_frequencies):
        song_info = all_song_info[song]
        artist_uri = song_info['artist_uri'].replace('spotify:artist:', '')
        if artist_uri not in artist_popularities:
            artist_batch.append(artist_uri)
        if len(artist_batch) == artist_batch_size:
            try:
                artists_info = spotify_client.make_api_request(f"artists?ids={','.join(artist_batch)}")
                for artist_info in artists_info["artists"]:
                    artist_popularities[artist_info["id"]] = artist_info["popularity"]
            except:
                print("Error in batch:", artist_batch)
            
            artist_batch = []
    
    # save artist_popularity to disk
    with open(filepath, "w") as f:
        json.dump(artist_popularities, f, indent=4)
            
    return artist_popularities


def plot_song_frequency_errors(song_frequencies, artist_popularities, scraped_streams, all_song_info):
    # Convert Song Counts to Ranks
    song_ranks = {perc: {song: rank + 1 for rank, (song, _) in enumerate(sorted(song_counts.items(), key=lambda x: x[1], reverse=True))} for perc, song_counts in song_frequencies.items()}

    #pdb.set_trace()

    # Normalize Ranks to Relative Ranks for Each Subset
    relative_ranks = {}
    for perc, ranks in song_ranks.items():
        total_songs_in_subset = len(ranks)
        relative_ranks[perc] = {song_id: rank / total_songs_in_subset for song_id, rank in ranks.items()}

    # Compute Rank Differences with Ground Truth
    rank_diffs = {perc: {song: relative_ranks[perc].get(song, 1) - relative_ranks[1].get(song, 1) for song in set(relative_ranks[perc])} for perc in relative_ranks if perc != 1}
    
    # do the same for artist popularities
    total_artist_popularities = len(artist_popularities)
    artist_popularities = {song: (rank + 1) / total_artist_popularities for rank, (song, _) in enumerate(sorted(artist_popularities.items(), key=lambda x: x[1], reverse=True))}
    #pdb.set_trace()
    # return large value if song not in artist_popularities, this is filtered out later before plotting
    rank_diffs_artist_popularities = {song: artist_popularities.get(all_song_info[song]["artist_uri"].replace("spotify:artist:", ""), 1000) - relative_ranks[1].get(song, 1) for song in set(relative_ranks[1])}


    # do the same for scraped streams
    scraped_streams = scraped_streams.dropna()
    scraped_streams = scraped_streams.sort_values(by='streams', ascending=False)
    total_scraped_streams = scraped_streams.shape[0]

    # Create the dictionary with the required calculation
    scraped_streams = {row.track_uri: (rank + 1) / total_scraped_streams for rank, row in enumerate(scraped_streams.itertuples())}
    rank_diffs_scraped_streams = {song: scraped_streams.get(all_song_info[int(song)]["track_uri"].replace("spotify:track:", ""), 1000) - relative_ranks[1].get(song, 1) for song in set(relative_ranks[1]) if all_song_info[int(song)]["track_uri"].replace("spotify:track:", "") in scraped_streams}

    # Prepare Data for Plotting
    data_for_plotting = []
    for perc, diffs in rank_diffs.items():
        for song, diff in diffs.items():
            if diff <= 1 and diff >=-1:
                data_for_plotting.append({'Source': f"{perc}% of dataset", 'Error': diff})
    
    for song, diff in rank_diffs_artist_popularities.items():
        if diff <= 1 and diff >=-1:
            data_for_plotting.append({'Source': f"Artist popularity (spotify API)", 'Error': diff})
    
    for song, diff in rank_diffs_scraped_streams.items():
        if diff <= 1 and diff >=-1:
            data_for_plotting.append({'Source': f"Song streams (scraped)", 'Error': diff})

    # Plot Error Distributions
    df = pd.DataFrame(data_for_plotting)
    plt.figure(figsize=(7, 4))
    sns.histplot(data=df, x='Error', hue='Source', fill=False, common_bins=True, stat='density', bins=100, element='step', alpha=0.6)
    plt.title('Relative Rank Differences of Songs')
    plt.xlabel('Relative Rank Difference')
    plt.ylabel('Density')
    plt.tight_layout()
    plt.savefig(f"spotify_scraper/relative_rank_differences.pdf")




if __name__ == "__main__":

    raw_path = "/fast/jbaumann/spotify-algo-collective-action/data/million_playlist_dataset/data/_original_dataset"
    nr_of_data_files = 10
    nr_of_data_files = -1

    percentages_of_data = [0.01, 0.1, 0.25, 0.5, 1]
    #percentages_of_data = [1]
    assert 1 in percentages_of_data, "1 must be in percentages_of_data"
    song_frequencies, all_song_info = get_song_frequencies(percentages_of_data, raw_path, nr_of_data_files)

    # Example usage
    # song_ids = [
    #     "7GX5flRQZVHRAGd6B4TmDO",
    #     "2EEeOnHehOozLq4aS0n6SL",
    #     "7yyRTcZmCiyzzJlNzGC9Ol",
    #     "3DXncPQOG4VBw3QHh3S817"
    #     "5dNfHmqgr128gMY2tc5CeJ",
    #     "4Km5HrUvYTaSUfiSGPJeQR",
    #     "0SGkqnVQo9KPytSri1H6cF",
    #     "5hTpBe8h35rJ67eAWHQsJx",
    #     "3a1lNhkSLSkpJE4MSHpDu9",
    #     "152lZdxL1OR0ZMW6KquMif"
    #     ]
    
    artist_popularities = get_artist_popularities(song_frequencies[1], all_song_info)
    
    scraped_streams = scrape_streams(song_frequencies[1], all_song_info)

    plot_song_frequency_errors(song_frequencies, artist_popularities, scraped_streams, all_song_info)

    #pdb.set_trace()
    print("Done")