import torch
import torch.nn.functional as F
import numpy as np
from src.rta.utils import padded_avg, get_device
from src.data_manager.data_manager import SequentialTrainDataset, pad_collate
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
import tqdm
import time
from torch import Tensor
from torch.nn.functional import log_softmax
import pdb
import matplotlib.pyplot as plt
import sys
from utils import disable_progress_bar

class RTAModel(torch.nn.Module):
    """ The main class for creating RTA models. Each consist fo the combination of a Representer with an Aggregator,
     which are jointly trained by iterating over the training set (using the DataManager)"""
    def __init__(self,
               data_manager,
               representer,
               aggregator,
               training_params = {}):
      super(RTAModel, self).__init__()
      self.data_manager = data_manager
      self.representer = representer
      self.aggregator = aggregator
      self.training_params = training_params

    def chose_negative_examples(self, X_pos_rep, x_neg, pad_mask):
        # Negative examples are partly made of hard negatives and easy random negatives
        X_neg_rep = self.representer(x_neg)
        easy_neg_rep = X_neg_rep[:,:self.training_params['n_easy'],...]

        # draw hard negatives using nearst neighbours in the first layer song embedding space
        X_rep_avg = padded_avg(X_pos_rep, ~pad_mask)
        neg_prods = torch.diagonal(X_neg_rep.matmul(X_rep_avg.T), dim1=2, dim2=0).T
        top_neg_indices = torch.topk(neg_prods, k=self.training_params['n_hard'], dim=1)[1]
        hard_indices = torch.gather(x_neg, 1, top_neg_indices)

        hard_neg_rep = self.representer((hard_indices))
        X_neg_final = torch.cat([easy_neg_rep, hard_neg_rep], dim=1)
        return X_neg_final

    def compute_pos_loss_batch(self, X_agg, Y_pos_rep, pad_mask):
        # The part of the loss that concerns positive examples
        pos_prod = torch.sum(X_agg * Y_pos_rep, axis = 2).unsqueeze(2)
        pos_loss = padded_avg(-F.logsigmoid(pos_prod), ~pad_mask).mean()
        return pos_loss

    def compute_neg_loss_batch(self, X_agg, X_neg_rep, pad_mask):
        # The part of the loss that concerns negative examples
        X_agg_mean = padded_avg(X_agg, ~pad_mask)
        neg_prod = X_neg_rep.matmul(X_agg_mean.transpose(0,1)).transpose(1,2).diagonal().transpose(0,1)
        neg_loss = torch.mean(-F.logsigmoid(-neg_prod))
        return neg_loss

    def compute_loss_batch(self, x_pos, x_neg):
        # Computes the entirety of the loss for a batch
        pad_mask = (x_pos == 0).to(get_device())

        X_pos_rep = self.representer(x_pos)
        input_rep = X_pos_rep[:,:-1,:] # take all elements of each sequence except the last
        Y_pos_rep = X_pos_rep[:,1:,:]  # take all elements of each sequence except the first

        X_agg = self.aggregator.aggregate(input_rep, pad_mask[:,:-1])
        X_neg_rep = self.chose_negative_examples(X_agg, x_neg, pad_mask[:,1:])

        pos_loss = self.compute_pos_loss_batch(X_agg, Y_pos_rep, pad_mask[:,1:])
        neg_loss = self.compute_neg_loss_batch(X_agg, X_neg_rep, pad_mask[:,1:])
        loss = pos_loss + neg_loss
        return loss

    def prepare_training_objects(self, tuning=False):
        # Prepare the optimizer, the scheduler and the data_loader that will be used for training
        optimizer = torch.optim.SGD(self.parameters(), lr= self.training_params['lr'], weight_decay=self.training_params['wd'], momentum=self.training_params['mom'], nesterov=self.training_params['nesterov'])
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, self.training_params['patience'], gamma=self.training_params['factor'], last_epoch=- 1, verbose=False)
        if tuning:
            train_indices = self.data_manager.train_indices
        else:
            train_indices = np.concatenate((self.data_manager.train_indices, self.data_manager.val_indices))
        train_dataset = SequentialTrainDataset(self.data_manager, train_indices, max_size=self.training_params['max_size'], n_neg=self.training_params['n_neg'])
        train_dataloader = DataLoader(train_dataset, batch_size = self.training_params['batch_size'], shuffle=True, collate_fn=pad_collate, num_workers=0)
        return optimizer, scheduler, train_dataloader

    def compute_recos(self, test_dataloader, n_recos=500, signal_as_track_id=-1, famous_song_id=-1):
        # Compute recommendations for playlist of the validation or test sel
        dev = get_device()
        n_p = len(test_dataloader.dataset)
        with torch.no_grad():
          self.eval()
        recos = np.zeros((n_p, n_recos))
        current_batch = 0
        all_rep = self.representer.compute_all_representations()

        # joe: Initialize variables to compute averages and count
        total_rank = []
        total_relative_rank = []
        count_signal_present = []
        count_famous_song_present = []
        total_contexts = 0

        print("Starting inference (not autoreg.)...")
        for X in tqdm.tqdm(test_dataloader, disable=disable_progress_bar, file=sys.stdout):

          X = X.long().to(dev)

          ## this replaces the last item of the seed contex with the famous song: # TODO: remove again, just for debugging
          #if True:
          #  X[:, -1] = famous_song_id

          bs = X.shape[0]
          seq_len = X.shape[1]
          X_rep = self.representer(X)
          X_agg = self.aggregator.aggregate_single(X_rep, torch.zeros((bs, seq_len)).to(dev))
          scores = X_agg.matmul(all_rep[1:-1].T)

          # joe additions

          if signal_as_track_id != -1:
            # Check if the signal is present in each item of the input sequence
            signal_present = (X == signal_as_track_id).any(dim=1)
            famous_song_present = (X == famous_song_id).any(dim=1)

            # Count the number of items where the signal is present
            count_signal_present.append(signal_present.sum().item())
            count_famous_song_present.append(famous_song_present.sum().item())

            # Compute the rank and relative rank of the signal
            sorted_scores, indices = torch.sort(scores, descending=True, dim=1)

            ranks = []
            signal_not_found=0
            for i in range(bs):
                try:
                    rank = (indices[i] == signal_as_track_id).nonzero(as_tuple=True)[0].item() + 1
                    ranks.append(rank)
                    total_contexts += 1
                except RuntimeError:
                    print(f"  tensor # {i}: Signal not found in the indices")
                    signal_not_found +=1
                    pass
            relative_ranks = [rank / scores.shape[1] for rank in ranks if rank]

            # Add the ranks and relative ranks to the total
            total_rank.extend(ranks)
            total_relative_rank.extend(relative_ranks)

            # joe additions end

          scores = scores.scatter(1, X.to(dev) - 1, value = - 10**3) # make sure songs in the seed are not recommended
          coded_recos = torch.topk(scores, k =n_recos, dim=1)[1].cpu().long()
          recos[current_batch * test_dataloader.batch_size: current_batch * test_dataloader.batch_size + bs] = coded_recos
          current_batch+=1
        self.train()

        if signal_as_track_id != -1:
          # Compute and print averages and count
          avg_rank = min(total_rank) # TODO: I changed this from np.mean(total_rank) to min(total_rank) to see how min rank changes with steps
          avg_relative_rank = np.mean(total_relative_rank)
          total_contexts = np.sum(total_contexts)
          count_signal_present = np.sum(count_signal_present)
          count_famous_song_present = np.sum(count_famous_song_present)
          #print(f"id={signal_as_track_id} / Contexts:{total_contexts} / Avg min rank per step: {avg_rank} / Average rel. rank: {avg_relative_rank} / # of times signal in seed: {count_signal_present}, famous song: {count_famous_song_present}")
          if signal_not_found >0:
            print(f"  Signal not found in {signal_not_found} tensors")
        else:
          avg_rank = None
          avg_relative_rank = None
          count_signal_present = None
          count_famous_song_present = None
          
        return recos, total_contexts, avg_rank, avg_relative_rank, count_signal_present, count_famous_song_present




    def run_training(self, tuning=False, savePath=False, autoregressive=0, signal=-1, outfile="recos", eval_after_each_epoch=False):

      # Initialize lists to store performance measures and signal ranks
      loss_values = []
      rprec_values = []
      ndcg_values = []
      click_values = []
      success_values = []
      avg_rank_values = []
      avg_relative_rank_values = []

      # get track ids for all track uris
      tracks_id_by_uri = {value['track_uri']: value['id'] for key, value in self.data_manager.tracks_info.items()}
      try:
        signal_as_track_id = tracks_id_by_uri[signal]
        print(">>>>>signal_as_track_id:", signal_as_track_id)
      except:
        signal_as_track_id = -1
      
      try:
        famous_song_id = tracks_id_by_uri["spotify:track:7yyRTcZmCiyzzJlNzGC9Ol"]
        print(">>>>>famous_song_id:", famous_song_id)
      except:
        famous_song_id = -1

      # Runs the training loop of the RTAModel
      if tuning :
        test_evaluator, test_dataloader = self.data_manager.get_test_data("val")
      else :
        test_evaluator, test_dataloader = self.data_manager.get_test_data("test")
      optimizer, scheduler, train_dataloader = self.prepare_training_objects(tuning)
      batch_ct = 0
      print_every = False
      if "step_every" in self.training_params.keys():
        print_every = True
      start = time.time()
      if savePath:
        torch.save(self, savePath)
      for epoch in range(self.training_params['n_epochs']):
        print("Epoch %d/%d" % (epoch,self.training_params['n_epochs']))
        print("Elapsed time : %.0f seconds" % (time.time() - start))
        for xx_pad, yy_pad_neg, x_lens in tqdm.tqdm(train_dataloader, disable=disable_progress_bar, file=sys.stdout):
          self.train()
          optimizer.zero_grad()
          loss = self.compute_loss_batch(xx_pad.to(get_device()), yy_pad_neg.to(get_device()))
          loss.backward()
          if self.training_params['clip'] :
            clip_grad_norm_(self.parameters(), max_norm=self.training_params['clip'], norm_type=2)
          optimizer.step()
          if print_every :
            if batch_ct % self.training_params["step_every"] == 0:
              scheduler.step()
              print(loss.item())

              # Append the loss value to the list
              loss_values.append(loss.item())

              if autoregressive > 0:
                # TODO: not used anymore
                # recos = self.compute_recos_autoregressive(test_dataloader, max_context_length=self.training_params['max_size'], signal_as_track_id=signal_as_track_id, famous_song_id=famous_song_id, autoregressive=autoregressive)
                pass
              else:
                recos, total_contexts, avg_rank, avg_relative_rank, count_signal_present, count_famous_song_present = self.compute_recos(test_dataloader, signal_as_track_id=signal_as_track_id, famous_song_id=famous_song_id)
                
                if signal_as_track_id != -1:
                  print(f"id={signal_as_track_id} / Contexts:{total_contexts} / Avg min rank per step: {avg_rank} / Average rel. rank: {avg_relative_rank} / # of times signal in seed: {count_signal_present}, famous song: {count_famous_song_present}")

                  # Append performance measures and signal ranks to the lists
                  avg_rank_values.append(avg_rank)
                  avg_relative_rank_values.append(avg_relative_rank)

              r_prec = test_evaluator.compute_all_R_precisions(recos)
              ndcg = test_evaluator.compute_all_ndcgs(recos)
              click = test_evaluator.compute_all_clicks(recos)
              if signal_as_track_id != -1:
                signal_recs = test_evaluator.compute_all_signal_recs(recos, signal_as_track_id)
                signal_recs_in_holdouts = test_evaluator.compute_all_signal_recs_in_holdouts(recos, signal_as_track_id)
                success_values.append(signal_recs_in_holdouts.sum())
                print("rprec : %.3f, ndcg : %.3f, click : %.3f, total_signal_recs : %d, total_signal_recs_in_holdouts : %d" % (r_prec.mean(), ndcg.mean(), click.mean(), signal_recs.sum(), signal_recs_in_holdouts.sum()))

              else:
                print("rprec : %.3f, ndcg : %.3f, click : %.3f" % (r_prec.mean(), ndcg.mean(), click.mean()))
              
              # Append performance measures and signal ranks to the lists
              rprec_values.append(r_prec.mean())
              ndcg_values.append(ndcg.mean())
              click_values.append(click.mean())

          batch_ct += 1
        if savePath:
          torch.save(self, savePath)

        if eval_after_each_epoch:
          np.save("%s" % (str(outfile) + f"_epoch{epoch}"), recos)
           

        base_path = str(self.data_manager.foldername).replace("resources/data", "")
        print("base_path ??? :", base_path)
        
        # Plot loss values
        plt.figure(figsize=(12, 8))
        plt.plot(loss_values, label='Loss')
        plt.xlabel('Steps')
        plt.ylabel('Loss')
        plt.legend()
        print("Loss values saved in %s/resources/models/%s/loss_values.png" % (base_path, self.data_manager.replication_folder_name))
        plt.savefig('%s/resources/models/%s/loss_values.png' % (base_path, self.data_manager.replication_folder_name))
        plt.close()
        plt.show()

        # Plot R-Precision
        plt.figure(figsize=(12, 8))
        plt.plot(rprec_values, label='R-Precision')
        plt.xlabel('Steps')
        plt.ylabel('R-Precision')
        plt.legend()
        plt.savefig('%s/resources/models/%s/rprec_values.png' % (base_path, self.data_manager.replication_folder_name))
        plt.close()

        # Plot NDCG
        plt.figure(figsize=(12, 8))
        plt.plot(ndcg_values, label='NDCG')
        plt.xlabel('Steps')
        plt.ylabel('NDCG')
        plt.legend()
        plt.savefig('%s/resources/models/%s/ndcg_values.png' % (base_path, self.data_manager.replication_folder_name))
        plt.close()

        # Plot Click
        plt.figure(figsize=(12, 8))
        plt.plot(click_values, label='Click')
        plt.xlabel('Steps')
        plt.ylabel('Click')
        plt.legend()
        plt.savefig('%s/resources/models/%s/click_values.png' % (base_path, self.data_manager.replication_folder_name))
        plt.close()

        if signal_as_track_id != -1:
          # Plot success values per step
          plt.figure(figsize=(12, 8))
          plt.plot(success_values, label='Success per step')
          plt.xlabel('Steps')
          plt.ylabel('Success per step')
          plt.legend()
          plt.savefig('%s/resources/models/%s/success_values.png' % (base_path, self.data_manager.replication_folder_name))
          plt.close()

          if autoregressive > 0:
            # Plot Avg min rank per step
            plt.figure(figsize=(12, 8))
            plt.plot(avg_rank_values, label='Avg min rank per step')
            plt.xlabel('Steps')
            plt.ylabel('Avg min rank per step')
            plt.legend()
            plt.savefig('%s/resources/models/%s/avg_rank_values.png' % (base_path, self.data_manager.replication_folder_name))
            plt.close()

            # Plot Average Relative Rank
            plt.figure(figsize=(12, 8))
            plt.plot(avg_relative_rank_values, label='Average Relative Rank')
            plt.xlabel('Steps')
            plt.ylabel('Average Relative Rank (%)')
            plt.legend()
            plt.savefig('%s/resources/models/%s/avg_relative_rank_values.png' % (base_path, self.data_manager.replication_folder_name))
            plt.close()
      return
