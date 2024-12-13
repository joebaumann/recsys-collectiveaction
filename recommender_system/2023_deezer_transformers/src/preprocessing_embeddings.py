import sys
#sys.path.append('') # TODO: only use for debugging
#sys.path.append('/home/jbaumann/spotify-algo-collective-action/recommender_system/2023_deezer_transformers') # TODO: only needed for debugging
import numpy as np
import tqdm
import argparse
from src.embeddings.model import MatrixFactorizationModel
from src.data_manager.data_manager import DataManager
from pathlib import Path
from utils import disable_progress_bar
import time
import psutil


def create_initial_embeddings(data_manager, embeddings_foldername):
    print("Creating initial song embeddings")
    #pdb.set_trace()
    mf_model = MatrixFactorizationModel(data_manager, foldername=embeddings_foldername, retrain=True, emb_size=128)
    print("finished song embeddings")
    return

def create_side_embeddings(data_manager):
    buckets_dur = data_manager.get_duration_bucket(data_manager.song_duration)
    buckets_pop = data_manager.get_pop_bucket(data_manager.song_pop)
    buckets_dur_dict = {i: [] for i in range(40)}
    buckets_pop_dict = {i: [] for i in range(100)}
    print("Creating duration buckets")
    for ind, b in enumerate(buckets_dur):
        buckets_dur_dict[b].append(ind)
    print("Creating popularity buckets")
    for ind, b in enumerate(buckets_pop):
        buckets_pop_dict[b].append(ind)

    print([len(v) for k,v in buckets_pop_dict.items()])
    # Create metadata initial embedding
    song_embeddings = np.load(data_manager.song_embeddings_path)
    print("Creating album embeddings")
    alb_embeddings = np.asarray([song_embeddings[data_manager.album_songs[i]].mean(axis=0) for i in tqdm.tqdm(range(len(data_manager.album_songs)), disable=disable_progress_bar, file=sys.stdout)])
    print("Creating artist embeddings")

    art_embeddings = np.asarray([song_embeddings[data_manager.artist_songs[i]].mean(axis=0) for i in tqdm.tqdm(range(len(data_manager.artist_songs)), disable=disable_progress_bar, file=sys.stdout)])

    pop_embeddings = np.asarray(
        [song_embeddings[buckets_pop_dict[i]].mean(axis=0) for i in tqdm.tqdm(range(len(buckets_pop_dict)), disable=disable_progress_bar, file=sys.stdout)])
    pop_embeddings[np.isnan(pop_embeddings)] = 0

    dur_embeddings = np.asarray(
        [song_embeddings[buckets_dur_dict[i]].mean(axis=0) for i in tqdm.tqdm(range(len(buckets_dur_dict)), disable=disable_progress_bar, file=sys.stdout)])

    np.save(data_manager.album_embeddings_path, alb_embeddings)
    np.save(data_manager.artist_embeddings_path, art_embeddings)
    np.save(data_manager.pop_embeddings_path, pop_embeddings)
    np.save(data_manager.dur_embeddings_path, dur_embeddings)

if __name__ == "__main__":
    print("deezer beginning of embeddings")
    start_time = time.perf_counter()
    parser = argparse.ArgumentParser()
    parser.add_argument("--mpd_path", type=str, required=False, default="../MPD/data",
                             help = "Path to MPD")
    parser.add_argument("--out_path", type=str, required=False, default="resources/data/rta_input",
                             help = "Path to rta input")

    parser.add_argument('-strategy', '--signal_planting_strategy', type=str, required=True)
    parser.add_argument('-b', '--budget', type=int, help='the number of signals to plant.')
    parser.add_argument('-s', '--seed', type=int, required=True, help='seed for random number generation.')
    parser.add_argument('-collr', '--collective_requirement', type=str, required=False, default="", help='empty string if collective is random subset or track uri if every collective playlist must contain this track.')

    parser.add_argument('--base_path', type=str, required=True)
    parser.add_argument('--base_path_data', type=str, required=True)
    parser.add_argument('--dataset_name', type=str, required=True)
    parser.add_argument('--dataset_size', type=str, required=True)
    parser.add_argument('--fold', type=str, required=True)
    parser.add_argument('--challenge_dataset_name', type=str, required=True)
    parser.add_argument('--outfile', type=str, required=True)
    parser.add_argument('--model_name', type=str, required=True)
    
    args = parser.parse_args()


    if "none" in args.signal_planting_strategy:
        manipulated_dataset_directory = args.signal_planting_strategy
    else:
        manipulated_dataset_directory = f"signal_planting_strategy_{args.signal_planting_strategy}_budget_{args.budget}{args.collective_requirement}"


    embeddings_foldername = Path(args.base_path_data, "recommender_system/2023_deezer_transformers/resources/data/embeddings", args.dataset_name, args.dataset_size, manipulated_dataset_directory, args.fold)
    embeddings_foldername.mkdir(parents=True, exist_ok=True)

    data_manager_path = str(Path(args.dataset_name, args.dataset_size, manipulated_dataset_directory, args.fold))

    foldername = str(Path(args.base_path_data, "recommender_system/2023_deezer_transformers/resources/data/"))

    data_manager = DataManager(replication_folder_name = data_manager_path, foldername = foldername, signal_planting_strategy=args.signal_planting_strategy)
    print(data_manager.binary_train_set)
    print(f"Total GB of RAM used (before embeddings): {psutil.virtual_memory().used / (1024**3)}")
    create_initial_embeddings(data_manager, embeddings_foldername)
    #create_side_embeddings(data_manager) # TODO: add again if needed for other song embedding representations

    print(f"Total GB of RAM used (after embeddings): {psutil.virtual_memory().used / (1024**3)}")

    print("end embeddings :):)")
    print("  >>>Finished preprocessing in {:.4f} seconds".format(time.perf_counter() - start_time))