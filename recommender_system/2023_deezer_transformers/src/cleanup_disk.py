import argparse
from pathlib import Path
import shutil



def cleanup_directory(directory):
    if directory.exists():
        print(f"  Removing directory: {directory} ...")
        shutil.rmtree(directory)
        print(f"  done.")
    else:
        print(f"  Directory not found: {directory}")


def cleanup_disk(args, manipulated_dataset_directory):
    
    out_path = Path(args.base_path_data, "recommender_system/2023_deezer_transformers", "resources/data/rta_input", args.dataset_name, args.dataset_size, manipulated_dataset_directory, args.fold)
    embeddings_foldername = Path(args.base_path_data, "recommender_system/2023_deezer_transformers", "resources/data/embeddings", args.dataset_name, args.dataset_size, manipulated_dataset_directory, args.fold)
    dataset_split_path = Path(args.base_path_data, "recommender_system/2023_deezer_transformers", "resources/data/dataset_split", args.dataset_name, args.dataset_size, manipulated_dataset_directory, args.fold)

    cleanup_directory(out_path)
    cleanup_directory(embeddings_foldername)
    cleanup_directory(dataset_split_path)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

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

    # unused arguments
    #parser.add_argument('--threads', type=int, required=False)
    parser.add_argument('--base_path_data', type=str, required=False)
    parser.add_argument('--dataset_name', type=str, required=False)

    args = parser.parse_args()


    if args.signal_planting_strategy == "none":
        manipulated_dataset_directory = "none"
    else:
        manipulated_dataset_directory = f"signal_planting_strategy_{args.signal_planting_strategy}_budget_{args.budget}{args.collective_requirement}"

    
    outfile_path = Path(args.base_path, "submissions_and_results", f"2023_deezer_{args.model_name}", args.dataset_size, manipulated_dataset_directory, args.fold)
    outfile = Path(outfile_path, f"{args.outfile}.npy")
    
    print("")
    
    print("cleaning disk...")
    cleanup_disk(args, manipulated_dataset_directory)
