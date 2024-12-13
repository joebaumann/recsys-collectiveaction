# Algorithmic Collective Action in Recommender Systems



This repository contains the code to reproduce the results of the NeurIPS 2024 paper [Algorithmic Collective Action in Recommender Systems: Promoting Songs by Reordering Playlists](https://arxiv.org/abs/2404.04269).

## Setup

Prepare the environment:
```
conda create -n collectiveaction python=3.9
conda activate collectiveaction
pip3 install torch torchvision torchaudio
conda install -c conda-forge implicit implicit-proc=*=gpu
pip install -r requirements.txt
```

### Data

Then download Spotify's Million Playlist Dataset (MPD) from [AIcrowd.com](https://www.aicrowd.com/challenges/spotify-million-playlist-dataset-challenge) and unzip it into the desired folder.

Note: You need to create an account and register to download the data.

## Model training

First, set the parameters in the /config.yaml file.

### Data Preprocessing and strategic collective action

Run the data preprocessing and the strategic training data manipulation as follows:
```
./preprocessing.sh none 1000 fold_0
```
where the command line parameters refer to
- the collective strategy: none, ???
- the collective budget, i.e., the number of playlists that should be manipulated: any number larger than 1
- the fold description: any unique string.


### Model training & evaluation
Run the data training and evaluation as follows:
```
./training_and_evaluation.sh none 1000 fold_0
```

### Hyperparameter robustness tests

To adjust the hyperparameters of the learned model, update the file `recommender_system/2023_deezer_transformers/resources/params/best_params_rta.json`.


## Citation

```bib
@inproceedings{baumann2024collectiveaction,
    title={Algorithmic Collective Action in Recommender Systems: Promoting Songs by Reordering Playlists},
    author={Baumann, Joachim and Mendler-D{\"u}nner, Celestine},
    booktitle = {Advances in Neural Information Processing Systems},
    year={2024},
    url={https://openreview.net/forum?id=wGjSbaMsop}
}
```
