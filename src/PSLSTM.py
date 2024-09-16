import torch
import numpy as np
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader
import torchmetrics
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from src.SyntheticDataGeneration import *
from src.utils import *

class PropensityScoreLSTM(torch.nn.Module):

    # LSTM to predict propensity scores

    def __init__(self, embedding_dim, hidden_dim, output_dim, vocab_size, num_layers):
        super(PropensityScoreLSTM, self).__init__()

        self.Embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.LSTM = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        self.FC = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, len_batch):
        x = self.Embedding(x) # batch_size * max_record_len * max_code_len * embedding_dim
        x = x.mean(2) # batch_size * max_record_len * embedding_dim
        x_packed = pack_padded_sequence(x, len_batch, batch_first=True, enforce_sorted=False)
        output, (_,_) = self.LSTM(x_packed)
        output_unpacked, len_batch_re = pad_packed_sequence(output, batch_first=True)

        return self.FC(output_unpacked), len_batch_re
    
def pad_collate_lstm(batch):
    
    # quality check done with keys-returning tensor match

    data, label, t_prob, outcome, data_len, keys = list(zip(*batch))
    data_len = list(data_len)
    
    padded_data = []
    padded_seqs = []
    
    for i in range(len(data)):
        padded_seqs.extend(data[i])

    padded_seqs = torch.nn.utils.rnn.pad_sequence(padded_seqs, batch_first=True, padding_value=0)
    
    prev_start = 0
    for i in range(len(data_len)):
        len_sum = int(np.sum(data_len[:i+1]))
        padded_data.append(padded_seqs[prev_start:len_sum])
        prev_start += data_len[i]
    
    padded_data = torch.nn.utils.rnn.pad_sequence(padded_data, batch_first=True, padding_value=0)
    
    return padded_data, torch.Tensor(label), torch.Tensor(t_prob), torch.Tensor(outcome), torch.Tensor(data_len)

def train_PSLSTM(training_dataset, validation_dataset, embedding_dim, hidden_dim, vocab_size, num_layers, 
                 training_epoch, batch_size, learning_rate, early_stop=True):
    
    print("loading data...")
    training_dataloader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True, collate_fn=pad_collate_lstm, drop_last=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=10, shuffle=True, collate_fn=pad_collate_lstm, drop_last=True)
    print("loading data done...")
 
    print("building and initializing model...")
    PSLSTM = PropensityScoreLSTM(embedding_dim, hidden_dim, 1, vocab_size, num_layers)
    Sigmoid = torch.nn.Sigmoid()
    BCE_loss_fn = nn.BCELoss()
    Optimizer = torch.optim.Adam(PSLSTM.parameters(), lr=learning_rate)
    print("building and initializing model done...")
    
    print("training starts...")
    training_loss_per_epoch = []
    validation_loss_per_epoch = []
    best_loss = np.inf
    best_model = None
    best_epoch = 0

    batch_loss_sum = 0.
    num_total_batch = len(validation_dataloader)
    PSLSTM.eval()
    for x_batch, label_batch, t_prob_batch, outcome_batch, len_batch in tqdm(validation_dataloader):

        with torch.no_grad():
            logit_batch, len_batch_re = PSLSTM(x_batch, len_batch)
            pred_batch = Sigmoid(logit_batch) # batch_size * max_record_len * 1
            pred_batch = pred_batch.reshape((pred_batch.shape[0], pred_batch.shape[1])) # batch_size * max_record_len
            last_pred_batch = pred_batch[torch.arange(10), len_batch_re-1]
            loss_batch = BCE_loss_fn(last_pred_batch, label_batch)
            batch_loss_sum += loss_batch.item()
        
    validation_loss = batch_loss_sum/num_total_batch
    print("initial validation loss: {}".format(validation_loss))

    for e in range(training_epoch):

        if early_stop:
            if (e+1-best_epoch) >= 5:
                # if the loss did not decrease for 5 epoch in a row, stop training
                break

        print("epoch {}".format(e+1))
        batch_loss_sum = 0.
        num_total_batch = len(training_dataloader)

        PSLSTM.train()
        for x_batch, label_batch, t_prob_batch, outcome_batch, len_batch in tqdm(training_dataloader):

            # for training loss and validation loss, binary treatment label is used with cross-entrophy loss
            # outcome is not used for training
            
            Optimizer.zero_grad()
            logit_batch, len_batch_re = PSLSTM(x_batch, len_batch)
            pred_batch = Sigmoid(logit_batch) # batch_size * max_record_len * 1
            pred_batch = pred_batch.reshape((pred_batch.shape[0], pred_batch.shape[1])) # batch_size * max_record_len
            last_pred_batch = pred_batch[torch.arange(batch_size), len_batch_re-1]
            loss_batch = BCE_loss_fn(last_pred_batch, label_batch)
            loss_batch.backward()
            Optimizer.step()

            batch_loss_sum += loss_batch.item()
        
        training_loss = batch_loss_sum/num_total_batch
        training_loss_per_epoch.append(training_loss)

        print("calculating validation loss...")
        batch_loss_sum = 0.
        num_total_batch = len(validation_dataloader)
        
        PSLSTM.eval()
        for x_batch, label_batch, t_prob_batch, outcome_batch, len_batch in tqdm(validation_dataloader):

            with torch.no_grad():
                logit_batch, len_batch_re = PSLSTM(x_batch, len_batch)
                pred_batch = Sigmoid(logit_batch) # batch_size * max_record_len * 1
                pred_batch = pred_batch.reshape((pred_batch.shape[0], pred_batch.shape[1])) # batch_size * max_record_len
                last_pred_batch = pred_batch[torch.arange(10), len_batch_re-1]
                loss_batch = BCE_loss_fn(last_pred_batch, label_batch)
                batch_loss_sum += loss_batch.item()
        
        validation_loss = batch_loss_sum/num_total_batch
        validation_loss_per_epoch.append(validation_loss)
        
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_epoch = e+1
            best_model = PSLSTM.state_dict()

        print("training loss: {}".format(training_loss))
        print("validation loss: {}".format(validation_loss))

    print("saving results...")
    result_dict = {"embedding_dim" : embedding_dim, "hidden_dim" : hidden_dim, "num_layers" : num_layers,
                   "training_epoch" : e+1, "batch_size" : batch_size, "learning_rate" : learning_rate, "vocab_size" : vocab_size,
                   "training_loss_per_epoch" : training_loss_per_epoch, "validation_loss_per_epoch" : validation_loss_per_epoch}

    PSLSTM.load_state_dict(best_model)

    return PSLSTM, result_dict

def evaluate_PSLSTM(testing_dataset, PSLSTM_model):
    
    testing_dataloader = DataLoader(testing_dataset, batch_size=10, shuffle=True, collate_fn=pad_collate_lstm, drop_last=True)
    Sigmoid = torch.nn.Sigmoid()
    MAE = torchmetrics.MeanAbsoluteError()
    weighted_MAE = torchmetrics.MeanAbsoluteError()
    
    print("evaluation starts...") # MAE, IPW-ATE
    y_z1_sum = 0.
    z1_sum = 0.
    y_z0_sum = 0.
    z0_sum = 0.
        
    PSLSTM_model.eval()
    for x_batch, label_batch, t_prob_batch, outcome_batch, len_batch in tqdm(testing_dataloader):

        # for evaluation, ground-truth treatment probability is used with mean absoulte error
        
        with torch.no_grad():           
            logit_batch, len_batch_re = PSLSTM_model(x_batch, len_batch)
            pred_batch = Sigmoid(logit_batch) # propensity score: batch_size * max_record_len * 1
            pred_batch = pred_batch.reshape((pred_batch.shape[0], pred_batch.shape[1])) # batch_size * max_record_len
            last_pred_batch = pred_batch[torch.arange(10), len_batch_re-1]
            MAE.update(last_pred_batch, t_prob_batch)
            weighted_MAE.update(t_prob_batch * last_pred_batch, t_prob_batch * t_prob_batch)

            # compute IPW-ATE
            y_z1, z1, y_z0, z0 = compute_IPW_ATE_partialsum(label_batch, outcome_batch, last_pred_batch)
            y_z1_sum += y_z1.item()
            z1_sum += z1.item()
            y_z0_sum += y_z0.item()
            z0_sum += z0.item()

    ipw_ate = (y_z1_sum-y_z0_sum) / (z1_sum+z0_sum)
    print("evaluation done...")
    return MAE.compute().item(), weighted_MAE.compute().item(), ipw_ate

def evaluate_PSLSTM_clip(testing_dataset, PSLSTM_model, clip_value):
    
    testing_dataloader = DataLoader(testing_dataset, batch_size=10, shuffle=True, collate_fn=pad_collate_lstm, drop_last=True)
    # (embedding_dim, hidden_dim, 1, vocab_size+1, num_layers)
    Sigmoid = torch.nn.Sigmoid()
    MAE = torchmetrics.MeanAbsoluteError()
    
    print("evaluation starts...") # MAE, IPW-ATE
    y_z1_sum = 0.
    z1_sum = 0.
    y_z0_sum = 0.
    z0_sum = 0.
        
    PSLSTM_model.eval()
    for x_batch, label_batch, t_prob_batch, outcome_batch, len_batch in tqdm(testing_dataloader):

        # for evaluation, ground-truth treatment probability is used with mean absoulte error
            
        with torch.no_grad():           
            logit_batch, len_batch_re = PSLSTM_model(x_batch, len_batch)
            pred_batch = Sigmoid(logit_batch) # propensity score: batch_size * max_record_len * 1
            pred_batch = pred_batch.reshape((pred_batch.shape[0], pred_batch.shape[1])) # batch_size * max_record_len
            last_pred_batch = pred_batch[torch.arange(10), len_batch_re-1]
            last_pred_batch = torch.clip(last_pred_batch, min=clip_value[0], max=clip_value[1])

            MAE.update(last_pred_batch, t_prob_batch)

            # compute IPW-ATT
            y_z1, z1, y_z0, z0 = compute_IPW_ATE_partialsum(label_batch, outcome_batch, last_pred_batch)
            y_z1_sum += y_z1.item()
            z1_sum += z1.item()
            y_z0_sum += y_z0.item()
            z0_sum += z0.item()

    ipw_ate = (y_z1_sum-y_z0_sum) / (z1_sum+z0_sum)
    print("evaluation done...")
    return MAE.compute().item(), ipw_ate

def evaluate_PSLSTM_trimming(testing_dataset, PSLSTM_model, trimming_range):
    
    testing_dataloader = DataLoader(testing_dataset, batch_size=10, shuffle=True, collate_fn=pad_collate_lstm, drop_last=True)
    # (embedding_dim, hidden_dim, 1, vocab_size+1, num_layers)
    Sigmoid = torch.nn.Sigmoid()
    MAE = torchmetrics.MeanAbsoluteError()
    
    print("evaluation starts...")
    y_z1_sum = 0.
    z1_sum = 0.
    y_z0_sum = 0.
    z0_sum = 0.
        
    PSLSTM_model.eval()
    for x_batch, label_batch, t_prob_batch, outcome_batch, len_batch in tqdm(testing_dataloader):

        # for evaluation, ground-truth treatment probability is used with mean absoulte error
            
        with torch.no_grad():           
            logit_batch, len_batch_re = PSLSTM_model(x_batch, len_batch)
            pred_batch = Sigmoid(logit_batch) # propensity score: batch_size * max_record_len * 1
            pred_batch = pred_batch.reshape((pred_batch.shape[0], pred_batch.shape[1])) # batch_size * max_record_len
            last_pred_batch = pred_batch[torch.arange(10), len_batch_re-1]
            last_pred_batch = last_pred_batch.flatten()

            trimming_last_pred_batch = last_pred_batch[(last_pred_batch >= trimming_range[0]) & (last_pred_batch <= trimming_range[1])]
            t_prob_batch = t_prob_batch[(last_pred_batch >= trimming_range[0]) & (last_pred_batch <= trimming_range[1])]
            label_batch = label_batch[(last_pred_batch >= trimming_range[0]) & (last_pred_batch <= trimming_range[1])]
            outcome_batch = outcome_batch[(last_pred_batch >= trimming_range[0]) & (last_pred_batch <= trimming_range[1])]
            MAE.update(trimming_last_pred_batch, t_prob_batch)

            # compute IPW-ATE
            y_z1, z1, y_z0, z0 = compute_IPW_ATE_partialsum(label_batch, outcome_batch, trimming_last_pred_batch)
            y_z1_sum += y_z1.item()
            z1_sum += z1.item()
            y_z0_sum += y_z0.item()
            z0_sum += z0.item()

    ipw_ate = (y_z1_sum-y_z0_sum) / (z1_sum+z0_sum)
    print("evaluation done...")
    return MAE.compute().item(), ipw_ate

# need to revise nfold implementation

def nfold_train_PSLSTM(data_dir, file_name, saving_dir, nfold, embedding_dim, hidden_dim, vocab_size, num_layers, 
                     training_epoch, batch_size, learning_rate, early_stop):

    nfold_result_dict = dict()
    data_nfold = np.array(load_nfold_data(data_dir, file_name, nfold)) # list of single folded split
    nfold_idx = np.tile(np.arange(nfold),2)

    for n in range(nfold):

        validation_idx = nfold_idx[n]
        testing_idx = nfold_idx[n+1]
        training_idx = np.array(nfold_idx[n+2:n+nfold])

        training_data = merge_folds(data_nfold[training_idx])
        validation_data = data_nfold[validation_idx]
        testing_data = data_nfold[testing_idx]

        training_dataset = SyntheticDataset(training_data)
        validation_dataset = SyntheticDataset(validation_data)
        testing_dataset = SyntheticDataset(testing_data)

        best_model, result_dict = train_PSLSTM(training_dataset, validation_dataset, 
                                               embedding_dim, hidden_dim, vocab_size, num_layers,
                                             training_epoch, batch_size, learning_rate, early_stop=early_stop)
        
        ps_mae, weighted_ps_mae, ipw_ate = evaluate_PSLSTM(testing_dataset, best_model)
        print("IPW-ATE : {}".format(ipw_ate))
        print("PS-MAE : {}".format(ps_mae))
        print("weighted PS-MAE : {}".format(weighted_ps_mae))

        nfold_result_dict[n] = copy.deepcopy(result_dict)
        
        torch.save(best_model.state_dict(), saving_dir+"PSLSTM_fold{}_model.pt".format(n))

    save_data(saving_dir+"PSLSTM_result_dict.pkl", nfold_result_dict)

def nfold_evaluate_PSLSTM(data_dir, file_name, model_dir, nfold, embedding_dim, hidden_dim, vocab_size, num_layers, 
                          trimming_range, clip_range):

    nfold_evaluation_dict = dict()
    data_nfold = np.array(load_nfold_data(data_dir, file_name, nfold)) # list of single folded split
    nfold_idx = np.tile(np.arange(nfold),2)

    for n in range(nfold):

        testing_idx = nfold_idx[n+1]
        testing_data = data_nfold[testing_idx]
        testing_dataset = SyntheticDataset(testing_data)

        PSLSTM = PropensityScoreLSTM(embedding_dim, hidden_dim, 1, vocab_size, num_layers)
        model = model_dir + "PSLSTM_fold{}_model.pt".format(n)
        PSLSTM.load_state_dict(torch.load(model))

        ps_mae, weighted_ps_mae, ipw_ate = evaluate_PSLSTM(testing_dataset, PSLSTM)
        print("IPW-ATE : {}".format(ipw_ate))
        print("PS-MAE : {}".format(ps_mae))
        print("weighted PS-MAE : {}".format(weighted_ps_mae))

        if trimming_range:
            trimming_ps_mae, trimming_ipw_ate = evaluate_PSLSTM_trimming(testing_dataset, PSLSTM, trimming_range)
            print("IPW-ATE with trimming: {}".format(trimming_ipw_ate))
            print("PS-MAE with trimming: {}".format(trimming_ps_mae))

        if clip_range:
            clip_ps_mae, clip_ipw_ate = evaluate_PSLSTM_clip(testing_dataset, PSLSTM, clip_range)
            print("IPW-ATE with clip: {}".format(clip_ipw_ate))
            print("PS-MAE with clip: {}".format(clip_ps_mae))

        nfold_evaluation_dict[n] = {"PS_MAE" : ps_mae, "IPW_ATE" : ipw_ate, "weighted PS_MAE" : weighted_ps_mae}
        
        if trimming_range:
            nfold_evaluation_dict[n].update({"trimming_PS_MAE" : trimming_ps_mae, "trimming_IPW_ATE" : trimming_ipw_ate})

        if clip_range:
            nfold_evaluation_dict[n].update({"clip_PS_MAE" : clip_ps_mae, "clip_IPW_ATE" : clip_ipw_ate})

    return nfold_evaluation_dict

def tune_batch_size_PSLSTM(data_dir, file_name, saving_dir, batch_size_list, nfold, embedding_dim, hidden_dim, vocab_size, num_layers, 
                           training_epoch, learning_rate, early_stop):
    
    tuning_result_dict = dict()
    data_nfold = np.array(load_nfold_data(data_dir, file_name, nfold)) # list of single folded split
    training_data = merge_folds(data_nfold[:-2])
    validation_data = data_nfold[-1]
    
    for batch_size in batch_size_list:
        
        print("building and initializing model...")
        training_dataset = SyntheticDataset(training_data)
        validation_dataset = SyntheticDataset(validation_data)

        best_model, result_dict = train_PSLSTM(training_dataset, validation_dataset, embedding_dim, hidden_dim, vocab_size, 
                                               num_layers, training_epoch, batch_size, learning_rate, early_stop)

        tuning_result_dict[batch_size] = {"training_loss_per_epoch" : copy.deepcopy(result_dict["training_loss_per_epoch"]),
                                          "validation_loss_per_epoch" : copy.deepcopy(result_dict["validation_loss_per_epoch"])}
    
    print("saving result...")
    saving = saving_dir + "PSLSTM_batch_size_tuning.pkl"
    save_data(saving, tuning_result_dict)

def get_propensity_score_PSLSTM(testing_dataset, PSLSTM_model):

    testing_dataloader = DataLoader(testing_dataset, batch_size=10, shuffle=True, collate_fn=pad_collate_lstm, drop_last=True)

    Sigmoid = torch.nn.Sigmoid()
    ps_list = []
    t_prob_list = []

    PSLSTM_model.eval()
    for x_batch, label_batch, t_prob_batch, outcome_batch, len_batch in tqdm(testing_dataloader):
            
        with torch.no_grad():           
            logit_batch, len_batch_re = PSLSTM_model(x_batch, len_batch)
            pred_batch = Sigmoid(logit_batch) # propensity score: batch_size * max_record_len * 1
            pred_batch = pred_batch.reshape((pred_batch.shape[0], pred_batch.shape[1])) # batch_size * max_record_len
            last_pred_batch = pred_batch[torch.arange(10), len_batch_re-1]
            last_pred_batch = last_pred_batch.flatten()
            
            ps_list.extend(last_pred_batch.tolist())
            t_prob_list.extend(t_prob_batch.tolist())
            
    return ps_list, t_prob_list

def nfold_plot_propensity_score_PSLSTM(data_dir, file_name, model_dir, nfold, embedding_dim, hidden_dim, vocab_size, num_layers):

    data_nfold = np.array(load_nfold_data(data_dir, file_name, nfold)) # list of single folded split
    nfold_idx = np.tile(np.arange(nfold),2)
    
    for n in range(nfold):

        testing_idx = nfold_idx[n+1]
        testing_data = data_nfold[testing_idx]
        testing_dataset = SyntheticDataset(testing_data)

        PSLSTM = PropensityScoreLSTM(embedding_dim, hidden_dim, 1, vocab_size, num_layers)
        model = model_dir + "PSLSTM_fold{}_model.pt".format(n)
        PSLSTM.load_state_dict(torch.load(model))
        
        ps_list, t_prob_list = get_propensity_score_PSLSTM(testing_dataset, PSLSTM)
        fig, (ax1, ax2) = plt.subplots(1, 2)
        sns.histplot(ps_list, kde=True, ax=ax1)
        ax1.set_xlim(0,1)
        ax1.set_xlabel("estimated propensity score")
        sns.histplot(t_prob_list, kde=True, ax=ax2)
        ax2.set_xlim(0,1)
        ax2.set_xlabel("true propensity score")
        plt.show()
