#!/bin/bash

# make bash script executable: chmod +x preprocessing.sh

python=... # python executable
base_path_data=... # path to the data directory
base_path=... # path to this repository
dataset_name="million_playlist_dataset"

dataset_size="data"
#dataset_size="data_small"
#dataset_size="data_very_small"

strategy=$1 # e.g. none, random, ...
budget=$2 # e.g., 50, 100, ...
fold=$3 # e.g., fold_0, fold_1, ...

challenge_dataset_name="challenge_dataset"
outfile="submission.csv"
model_type="MF-Transformer"
algo="2023_deezer_$model_type"

cd recommender_system/2023_deezer_transformers

cmd_line_args="-strategy $strategy --budget $budget --seed 42 --base_path $base_path --base_path_data $base_path_data --dataset_name $dataset_name --dataset_size $dataset_size --fold $fold --challenge_dataset_name $challenge_dataset_name --model_name $model_type --outfile $outfile"

if [ "$strategy" == "none" ]
then
    manipulated_dataset_directory="none"
else
    manipulated_dataset_directory="signal_planting_strategy_${strategy}_budget_${budget}"
fi
submission_filepath="$base_path/submissions_and_results/2023_deezer_$model_type/$dataset_size/$manipulated_dataset_directory/$fold/submission.csv"

# preprocessing the embeddings after preprocessing_and_collective_action.py has already been run
cmd="-m src.preprocessing_and_collective_action $cmd_line_args"
echo -e "\n>>Running: $python $cmd \n"
eval $python $cmd

echo "Done with preprocessing!"
