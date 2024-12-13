import json, argparse, os, time, datetime

from src.data_manager.data_manager import DataManager

from src.rta.utils import get_device
from src.rta.rta_model import RTAModel
from src.rta.aggregator.gru import GRUNet
from src.rta.aggregator.cnn import GatedCNN
from src.rta.aggregator.decoder import DecoderModel
from src.rta.aggregator.base import AggregatorBase
from src.rta.representer.base_representer import BaseEmbeddingRepresenter
from src.rta.representer.fm_representer import FMRepresenter
from src.rta.representer.attention_representer import AttentionFMRepresenter
import numpy as np
from pathlib import Path
from utils import get_target_song_info

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--model_name", type = str, required = True,
                    help = "Name of model to train")
  parser.add_argument("--params_file", type = str, required = False,
                    help = "File for hyperparameters", default = "resources/params/best_params_rta.json")
  

  parser.add_argument('-strategy', '--signal_planting_strategy', type=str, required=True)
  parser.add_argument('-b', '--budget', type=int, help='the number of signals to plant.')
  parser.add_argument('-s', '--seed', type=int, required=False, help='seed for random number generation.')
  parser.add_argument('--base_path', type=str, required=True)
  parser.add_argument('--base_path_data', type=str, required=True)
  parser.add_argument('--dataset_name', type=str, required=True)
  parser.add_argument('--dataset_size', type=str, required=True)
  parser.add_argument('--fold', type=str, required=True)
  parser.add_argument('--challenge_dataset_name', type=str, required=True)
  parser.add_argument('--outfile', type=str, required=True)


  args = parser.parse_args()
  
  print("running main...")

  if "none" in args.signal_planting_strategy:
      manipulated_dataset_directory = args.signal_planting_strategy
  else:
      manipulated_dataset_directory = f"signal_planting_strategy_{args.signal_planting_strategy}_budget_{args.budget}"

  signal_track_ids, track_id_by_track_uri, additionaltargetsong = get_target_song_info(args.signal_planting_strategy)
  signals = list(track_id_by_track_uri.keys())

  
  models_path = Path(args.base_path_data, "recommender_system/2023_deezer_transformers", "resources/models", args.dataset_name, args.dataset_size, manipulated_dataset_directory, args.fold)
  models_path.mkdir(parents=True, exist_ok=True)
  models_path = str(models_path)

  outfile_path = Path(args.base_path, "submissions_and_results", f"2023_deezer_{args.model_name}", args.dataset_size, manipulated_dataset_directory, args.fold)
  outfile_path.mkdir(parents=True, exist_ok=True)
  outfile = str(Path(outfile_path, args.outfile))

  data_manager_path = str(Path(args.dataset_name, args.dataset_size, manipulated_dataset_directory, args.fold))

  foldername = str(Path(args.base_path_data, "recommender_system/2023_deezer_transformers/resources/data/"))

  data_manager = DataManager(replication_folder_name = data_manager_path, foldername = foldername, signal_planting_strategy=args.signal_planting_strategy)

  params_filepath = str(Path(args.base_path, "recommender_system/2023_deezer_transformers", args.params_file))
  with open(params_filepath, "r") as f:
    p = json.load(f)

  tr_params = p[args.model_name]
  #tr_params['n_epochs'] = 1 # TODO: just for debugging!!! remove after!!!
  if args.model_name == "MF-GRU":
    print("Initialize Embeddings")
    representer = BaseEmbeddingRepresenter(data_manager, tr_params['d'])
    print("Initialize GRU")
    aggregator = GRUNet(tr_params['d'], tr_params['h_dim'], tr_params['d'], tr_params['n_layers'], tr_params['drop_p'])

  if args.model_name == "MF-CNN":
    print("Initialize Embeddings")
    representer = BaseEmbeddingRepresenter(data_manager, tr_params['d'])
    print("Initialize Gated-CNN")
    aggregator = GatedCNN(tr_params['d'], tr_params['n_layers'], tr_params['kernel_size'], tr_params['conv_size'], tr_params['res_block_count'], k_pool=tr_params['k_pool'], drop_p=tr_params['drop_p']).to(get_device())

  if args.model_name == "MF-AVG":
    print("Initialize Embeddings")
    representer = BaseEmbeddingRepresenter(data_manager, tr_params['d'])
    print("Initialize vanilla matrix factorization")
    aggregator = AggregatorBase()

  if args.model_name == "MF-Transformer":
    print("Initialize Embeddings")
    representer = BaseEmbeddingRepresenter(data_manager, tr_params['d'])
    print("Initialize Decoder")
    aggregator = DecoderModel(embd_size=tr_params["d"], max_len=tr_params["max_size"], n_head=tr_params["n_heads"], n_layers=tr_params["n_layers"], drop_p=tr_params["drop_p"])

  if args.model_name == "FM-Transformer":
    print("Initialize Embeddings")
    representer = FMRepresenter(data_manager, tr_params['d'])
    print("Initialize Decoder")
    aggregator = DecoderModel(embd_size=tr_params["d"], max_len=tr_params["max_size"], n_head=tr_params["n_heads"], n_layers=tr_params["n_layers"], drop_p=tr_params["drop_p"])

  if args.model_name == "NN-Transformer":
    print("Initialize Embeddings")
    representer = AttentionFMRepresenter(data_manager, emb_dim=tr_params['d'], n_att_heads=tr_params['n_att_heads'], n_att_layers=tr_params["n_att_layers"], dropout_att=tr_params["drop_att"])
    print("Initialize Decoder")
    aggregator = DecoderModel(embd_size=tr_params["d"], max_len=tr_params["max_size"], n_head=tr_params["n_heads"], n_layers=tr_params["n_layers"], drop_p=tr_params["drop_p"])

  rta_model = RTAModel(data_manager, representer, aggregator, training_params = tr_params).to(get_device())
  print("Train model %s" % args.model_name)
  savePath = "%s/%s" % (models_path, args.model_name)
  start_fit = time.time()
  #rta_model.run_training(tuning=False, savePath=savePath, outfile=outfile)
  #rta_model.run_training(tuning=False, savePath=savePath, signal=signals[0], outfile=outfile) # SAVE MODEL
  rta_model.run_training(tuning=False, savePath=False, signal=signals[0], outfile=outfile) # do not save model
  # TODO: make this a parameter
  end_fit = time.time()
  print("Model %s trained in %s " % (args.model_name, str(end_fit - start_fit)))
  test_evaluator, test_dataloader = data_manager.get_test_data("test")


  # get track ids for all track uris
  tracks_id_by_uri = {value['track_uri']: value['id'] for key, value in data_manager.tracks_info.items()}
  signal_as_track_ids = []
  try:
    for signal in signals:
      signal_as_track_id = tracks_id_by_uri[signal]
      signal_as_track_ids.append(signal_as_track_id)
  except:
    signal_as_track_ids = [-1]
  try:
    famous_song_id = tracks_id_by_uri["spotify:track:7yyRTcZmCiyzzJlNzGC9Ol"]
  except:
    famous_song_id = -1
  

  recos, total_contexts, avg_rank, avg_relative_rank, count_signal_present, count_famous_song_present = rta_model.compute_recos(test_dataloader, signal_as_track_id=signal_as_track_ids[0], famous_song_id=famous_song_id)
  end_predict = time.time()
  print("Model %s inferred in %s " % (args.model_name, str(end_predict - end_fit)))
  #os.makedirs(args.recos_path, exist_ok=True)
  #np.save("%s/%s" % (args.recos_path, args.model_name), recos)
  np.save("%s" % (outfile), recos)

  print(f" DONE!!! Contexts:{total_contexts} / Avg min rank per step: {avg_rank} / Average rel. rank: {avg_relative_rank} / # of times signal in seed: {count_signal_present}, famous song: {count_famous_song_present}")


  r_prec = test_evaluator.compute_all_R_precisions(recos)
  ndcg = test_evaluator.compute_all_ndcgs(recos)
  click = test_evaluator.compute_all_clicks(recos)
  if signal_as_track_ids != [-1]:
    for signal_as_track_id in signal_as_track_ids:
      print("---- check signal_as_track_id:", signal_as_track_id, "----")
      signal_recs = test_evaluator.compute_all_signal_recs(recos, signal_as_track_id)
      signal_recs_in_holdouts = test_evaluator.compute_all_signal_recs_in_holdouts(recos, signal_as_track_id)
      print("    rprec : %.3f, ndcg : %.3f, click : %.3f, total_signal_recs : %d, total_signal_recs_in_holdouts : %d" % (r_prec.mean(), ndcg.mean(), click.mean(), signal_recs.sum(), signal_recs_in_holdouts.sum()))
  else:
    print("rprec : %.3f, ndcg : %.3f, click : %.3f" % (r_prec.mean(), ndcg.mean(), click.mean()))
