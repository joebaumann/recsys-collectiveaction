import sys
import os
import logging
import json
import time
import seaborn as sns
import colorcet as cc
from tqdm import tqdm

def timer(func):
    """Decorator that prints the runtime of the decorated function"""

    def wrapper_timer(*args, **kwargs):
        start_time = time.perf_counter()
        value = func(*args, **kwargs)
        end_time = time.perf_counter()
        run_time = end_time - start_time
        timer_end_message = f"  ## timer ## Finished {func.__name__!r} in {run_time:.4f} seconds"
        #if "case" in kwargs:
        #    timer_end_message += f" / SCENARIO={kwargs['case']}"
        #if "seed" in kwargs:
        #    timer_end_message += f" / SEED={kwargs['seed']}"
        #if "path" in kwargs:
        #    timer_end_message += f" / Results stored in {kwargs['path']}"
        logging.debug(timer_end_message)
        print(timer_end_message)
        return value

    return wrapper_timer


@timer
def load_playlist_data(path, only_track_uris):
    all_playlists_only_track_uris = []
    all_playlists = []
    filenames = os.listdir(path)
    for filename in tqdm(sorted(filenames), file=sys.stdout):
        if filename.startswith("mpd.slice.") and filename.endswith(".json"):
            fullpath = os.sep.join((path, filename))
            f = open(fullpath)
            js = f.read()
            f.close()
            mpd_slice = json.loads(js)
            playlists_one_file = mpd_slice["playlists"]

            all_playlists_only_track_uris.extend([[track['track_uri'] for track in playlist['tracks']] for playlist in playlists_one_file])
            if not only_track_uris:
                all_playlists.extend(playlists_one_file)
    
    if only_track_uris:
        return all_playlists_only_track_uris
    else:
        return all_playlists_only_track_uris, all_playlists


def set_plot_style():
    sns.set_context("paper")
    sns.set(font='serif')
    sns.set_style("white", {
        "font.family": "serif",
        "font.serif": ["Times", "Palatino", "serif"]
    })



def start_logger(config, sanitized_date, logging_filepath):
    """
    Args:
        config (dict): the whole config file
        sanitized_date (str): sanitized start time for pipeline
    """

    logging.basicConfig(filename=f'{logging_filepath}/{config.logging_filename}-{sanitized_date}.log',
                        encoding='utf-8', format='%(levelname)s:%(message)s', level=logging.DEBUG)
    logging.captureWarnings(True)

    # ignore matplotlib output because it's overwhelming
    for name, logger in logging.root.manager.loggerDict.items():
        if name.startswith('matplotlib'):
            logger.disabled = True

    logging.debug(f'\n\nCONFIG: {str(dict(config))}\n\n')