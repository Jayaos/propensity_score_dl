import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
import torchmetrics
import pickle
import os
import copy
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from src.SyntheticDataGeneration import *
from src.utils import *

class PropensityScoreLogisticRegression(torch.nn.Module):

    def __init__(self, output_dim, vocab_size):
        super(PropensityScoreLogisticRegression, self).__init__()

        self.FC = nn.Linear(vocab_size, output_dim) 

    def forward(self, x):

        return self.FC(x) # batch_size * 1
    
def pad_collate_LR(batch):

    data, label, t_prob, outcome, data_len, keys = list(zip(*batch))
    
    return torch.stack(data), torch.Tensor(label), torch.Tensor(t_prob), torch.Tensor(outcome), torch.Tensor(data_len)

def train_PSLR(training_dataset, validation_dataset, vocab_size, training_epoch, batch_size, learning_rate, early_stop=True):
    
    print("loading data...")
    training_dataloader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True, collate_fn=pad_collate_LR, drop_last=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=10, shuffle=True, collate_fn=pad_collate_LR, drop_last=True)
    print("loading data done...")
 
    print("building and initializing model...")
    PSLR = PropensityScoreLogisticRegression(1, vocab_size)
    Sigmoid = torch.nn.Sigmoid()
    BCE_loss_fn = nn.BCELoss()
    Optimizer = torch.optim.Adam(PSLR.parameters(), lr=learning_rate)
    print("building and initializing model done...")
    
    print("training starts...")
    training_loss_per_epoch = []
    validation_loss_per_epoch = []
    best_loss = np.inf
    best_model = None
    best_epoch = 0

    PSLR.eval()
    batch_loss_sum = 0.
    num_total_batch = len(validation_dataloader)
    for x_batch, label_batch, t_prob_batch, outcome_batch, len_batch in tqdm(validation_dataloader):

        with torch.no_grad():
            logit_batch = PSLR(x_batch)
            pred_batch = Sigmoid(logit_batch) # batch_size * 1
            pred_batch = torch.reshape(pred_batch, (-1,))
            loss_batch = BCE_loss_fn(pred_batch, label_batch)
            batch_loss_sum += loss_batch.item()
        
    validation_loss = batch_loss_sum/num_total_batch
    print("initial validation loss: {}".format(validation_loss))

    PSLR.train()
    for e in range(training_epoch):

        if early_stop:
            if (e+1-best_epoch) >= 5:
                # if the loss did not decrease for 5 epoch in a row, stop training
                break

        print("epoch {}".format(e+1))
        batch_loss_sum = 0.
        num_total_batch = len(training_dataloader)

        for x_batch, label_batch, t_prob_batch, outcome_batch, len_batch in tqdm(training_dataloader):

            # for training loss and validation loss, binary treatment label is used with cross-entrophy loss
            # outcome is not used for training
            
            Optimizer.zero_grad()
            logit_batch = PSLR(x_batch)
            pred_batch = Sigmoid(logit_batch) # batch_size * 1
            pred_batch = torch.reshape(pred_batch, (-1,))
            loss_batch = BCE_loss_fn(pred_batch, label_batch)
            loss_batch.backward()
            Optimizer.step()

            batch_loss_sum += loss_batch.item()
        
        training_loss = batch_loss_sum/num_total_batch
        training_loss_per_epoch.append(training_loss)

        print("calculating validation loss...")
        batch_loss_sum = 0.
        num_total_batch = len(validation_dataloader)
        
        PSLR.eval()
        for x_batch, label_batch, t_prob_batch, outcome_batch, len_batch in tqdm(validation_dataloader):

            with torch.no_grad():
                logit_batch = PSLR(x_batch)
                pred_batch = Sigmoid(logit_batch) # batch_size * 1
                pred_batch = torch.reshape(pred_batch, (-1,))
                loss_batch = BCE_loss_fn(pred_batch, label_batch)
                batch_loss_sum += loss_batch.item()
        
        validation_loss = batch_loss_sum/num_total_batch
        validation_loss_per_epoch.append(validation_loss)
        
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_epoch = e+1
            best_model = PSLR.state_dict()

        print("training loss: {}".format(training_loss))
        print("validation loss: {}".format(validation_loss))

    result_dict = {"training_epoch" : e+1, "batch_size" : batch_size, "learning_rate" : learning_rate, "vocab_size" : vocab_size,
                   "training_loss_per_epoch" : training_loss_per_epoch, "validation_loss_per_epoch" : validation_loss_per_epoch,
                   "best_epoch" : best_epoch}
    
    PSLR.load_state_dict(best_model)

    return PSLR, result_dict

def evaluate_PSLR(testing_dataset, PSLR_model):
    
    testing_dataloader = DataLoader(testing_dataset, batch_size=10, shuffle=True, collate_fn=pad_collate_LR, drop_last=True)

    Sigmoid = torch.nn.Sigmoid()
    MAE = torchmetrics.MeanAbsoluteError()
    weighted_MAE = torchmetrics.MeanAbsoluteError()
    
    print("evaluation starts...") 
    # MAE, IPW-ATE
    y_z1_sum = 0.
    z1_sum = 0.
    y_z0_sum = 0.
    z0_sum = 0.
        
    PSLR_model.eval()
    for x_batch, label_batch, t_prob_batch, outcome_batch, len_batch in tqdm(testing_dataloader):

        # for evaluation, ground-truth treatment probability is used with mean absoulte error
            
        with torch.no_grad():           
            logit_batch = PSLR_model(x_batch)
            pred_batch = Sigmoid(logit_batch) # propensity score: batch_size * 1
            pred_batch = torch.reshape(pred_batch, (-1,))
            MAE.update(pred_batch, t_prob_batch)
            weighted_MAE.update(t_prob_batch * pred_batch, t_prob_batch * t_prob_batch)

            # compute IPW-ATE
            y_z1, z1, y_z0, z0 = compute_IPW_ATE_partialsum(label_batch, outcome_batch, pred_batch)
            y_z1_sum += y_z1.item()
            z1_sum += z1.item()
            y_z0_sum += y_z0.item()
            z0_sum += z0.item()

    ipw_ate = (y_z1_sum-y_z0_sum) / (z1_sum+z0_sum)
    return MAE.compute().item(), weighted_MAE.compute().item(), ipw_ate

def evaluate_PSLR_clip(testing_dataset, PSLR_model, clip_value):
    
    testing_dataloader = DataLoader(testing_dataset, batch_size=10, shuffle=True, collate_fn=pad_collate_LR, drop_last=True)

    Sigmoid = torch.nn.Sigmoid()
    MAE = torchmetrics.MeanAbsoluteError()
    
    print("evaluation starts...") 
    # MAE, IPW-ATE
    y_z1_sum = 0.
    z1_sum = 0.
    y_z0_sum = 0.
    z0_sum = 0.
        
    PSLR_model.eval()
    for x_batch, label_batch, t_prob_batch, outcome_batch, len_batch in tqdm(testing_dataloader):

        # for evaluation, ground-truth treatment probability is used with mean absoulte error
            
        with torch.no_grad():           
            logit_batch = PSLR_model(x_batch)
            pred_batch = Sigmoid(logit_batch) # propensity score: batch_size * 1
            pred_batch = torch.reshape(pred_batch, (-1,))
            pred_batch = torch.clip(pred_batch, min=clip_value[0], max=clip_value[1])

            MAE.update(pred_batch, t_prob_batch)

            # compute IPW-ATE
            y_z1, z1, y_z0, z0 = compute_IPW_ATE_partialsum(label_batch, outcome_batch, pred_batch)
            y_z1_sum += y_z1.item()
            z1_sum += z1.item()
            y_z0_sum += y_z0.item()
            z0_sum += z0.item()

    ipw_ate = (y_z1_sum-y_z0_sum) / (z1_sum+z0_sum)
    return MAE.compute().item(), ipw_ate

def evaluate_PSLR_trimming(testing_dataset, PSLR_model, trimming_range):
    
    testing_dataloader = DataLoader(testing_dataset, batch_size=10, shuffle=True, collate_fn=pad_collate_LR, drop_last=True)

    Sigmoid = torch.nn.Sigmoid()
    MAE = torchmetrics.MeanAbsoluteError()
    
    print("evaluation starts...") 
    # MAE, IPW-ATE
    y_z1_sum = 0.
    z1_sum = 0.
    y_z0_sum = 0.
    z0_sum = 0.
        
    PSLR_model.eval()
    for x_batch, label_batch, t_prob_batch, outcome_batch, len_batch in tqdm(testing_dataloader):

        # for evaluation, ground-truth treatment probability is used with mean absoulte error
            
        with torch.no_grad():           
            logit_batch = PSLR_model(x_batch)
            pred_batch = Sigmoid(logit_batch) # propensity score: batch_size * 1
            pred_batch = torch.reshape(pred_batch, (-1,))

            trimming_pred_batch = pred_batch[(pred_batch >= trimming_range[0]) & (pred_batch <= trimming_range[1])]
            t_prob_batch = t_prob_batch[(pred_batch >= trimming_range[0]) & (pred_batch <= trimming_range[1])]
            label_batch = label_batch[(pred_batch >= trimming_range[0]) & (pred_batch <= trimming_range[1])]
            outcome_batch = outcome_batch[(pred_batch >= trimming_range[0]) & (pred_batch <= trimming_range[1])]
            MAE.update(trimming_pred_batch, t_prob_batch)

            # compute IPW-ATE
            y_z1, z1, y_z0, z0 = compute_IPW_ATE_partialsum(label_batch, outcome_batch, trimming_pred_batch)
            y_z1_sum += y_z1.item()
            z1_sum += z1.item()
            y_z0_sum += y_z0.item()
            z0_sum += z0.item()

    ipw_ate = (y_z1_sum-y_z0_sum) / (z1_sum+z0_sum)
    return MAE.compute().item(), ipw_ate

def nfold_train_PSLR(data_dir, file_name, saving_dir, nfold, vocab_size, training_epoch, batch_size, learning_rate,
                     early_stop, HDPS):

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

        if HDPS:
            training_data = convert_HDPS_data_record(training_data, vocab_size)
            validation_data = convert_HDPS_data_record(validation_data, vocab_size)
            testing_data = convert_HDPS_data_record(testing_data, vocab_size)

        training_dataset = SyntheticDatasetCollapsedLR(training_data)
        validation_dataset = SyntheticDatasetCollapsedLR(validation_data)
        testing_dataset = SyntheticDatasetCollapsedLR(testing_data)

        if HDPS:
            best_model, result_dict = train_PSLR(training_dataset, validation_dataset, vocab_size*3, 
                                                 training_epoch, batch_size, learning_rate, early_stop=early_stop)
        else:
            best_model, result_dict = train_PSLR(training_dataset, validation_dataset, vocab_size, 
                                                 training_epoch, batch_size, learning_rate, early_stop=early_stop)
            
        ps_mae, weighted_ps_mae, ipw_ate = evaluate_PSLR(testing_dataset, best_model)
        print("IPW-ATE : {}".format(ipw_ate))
        print("PS-MAE : {}".format(ps_mae))
        print("weighted PS-MAE : {}".format(weighted_ps_mae))

        nfold_result_dict[n] = copy.deepcopy(result_dict)

        torch.save(best_model.state_dict(), saving_dir+"PSLR_fold{}_model.pt".format(n))

    save_data(saving_dir+"PSLR_result_dict.pkl", nfold_result_dict)

def nfold_evaluate_PSLR(data_dir, file_name, model_dir, nfold, vocab_size, trimming_range, clip_range, HDPS):

    nfold_evaluation_dict = dict()
    data_nfold = np.array(load_nfold_data(data_dir, file_name, nfold)) # list of single folded split
    nfold_idx = np.tile(np.arange(nfold),2)

    for n in range(nfold):

        testing_idx = nfold_idx[n+1]
        testing_data = data_nfold[testing_idx]
        if HDPS:
            testing_data = convert_HDPS_data_record(testing_data, vocab_size)
        testing_dataset = SyntheticDatasetCollapsedLR(testing_data)

        if HDPS:
            PSLR = PropensityScoreLogisticRegression(1, vocab_size*3)
        else:
            PSLR = PropensityScoreLogisticRegression(1, vocab_size)

        model = model_dir + "PSLR_fold{}_model.pt".format(n)
        PSLR.load_state_dict(torch.load(model))
            
        ps_mae, weighted_ps_mae, ipw_ate = evaluate_PSLR(testing_dataset, PSLR)
        print("IPW-ATE : {}".format(ipw_ate))
        print("PS-MAE : {}".format(ps_mae))
        print("weighted PS-MAE : {}".format(weighted_ps_mae))

        if trimming_range:
            trimming_ps_mae, trimming_ipw_ate = evaluate_PSLR_trimming(testing_dataset, PSLR, trimming_range)
            print("IPW-ATE with trimming: {}".format(trimming_ipw_ate))
            print("PS-MAE with trimming: {}".format(trimming_ps_mae))

        if clip_range:
            clip_ps_mae, clip_ipw_ate = evaluate_PSLR_clip(testing_dataset, PSLR, clip_range)
            print("IPW-ATE with clip: {}".format(clip_ipw_ate))
            print("PS-MAE with clip: {}".format(clip_ps_mae))

        nfold_evaluation_dict[n] = {"PS_MAE" : ps_mae, "IPW_ATE" : ipw_ate, "weighted PS_MAE" : weighted_ps_mae}
        
        if trimming_range:
            nfold_evaluation_dict[n].update({"trimming_PS_MAE" : trimming_ps_mae, "trimming_IPW_ATE" : trimming_ipw_ate})

        if clip_range:
            nfold_evaluation_dict[n].update({"clip_PS_MAE" : clip_ps_mae, "clip_IPW_ATE" : clip_ipw_ate})

    return nfold_evaluation_dict

def get_propensity_score_PSLR(testing_dataset, PSLR_model):

    testing_dataloader = DataLoader(testing_dataset, batch_size=10, shuffle=True, collate_fn=pad_collate_LR, drop_last=True)

    Sigmoid = torch.nn.Sigmoid()
    ps_list = []
    t_prob_list = []

    PSLR_model.eval()
    for x_batch, label_batch, t_prob_batch, outcome_batch, len_batch in tqdm(testing_dataloader):
            
        with torch.no_grad():           
            logit_batch = PSLR_model(x_batch)
            pred_batch = Sigmoid(logit_batch) # propensity score: batch_size * 1
            pred_batch = pred_batch.flatten()
            
            ps_list.extend(pred_batch.tolist())
            t_prob_list.extend(t_prob_batch.tolist())
            
    return ps_list, t_prob_list

def nfold_plot_propensity_score_LR(data_dir, file_name, model_dir, nfold, vocab_size, HDPS):

    data_nfold = np.array(load_nfold_data(data_dir, file_name, nfold)) # list of single folded split
    nfold_idx = np.tile(np.arange(nfold),2)
    
    for n in range(nfold):

        testing_idx = nfold_idx[n+1]
        testing_data = data_nfold[testing_idx]
        if HDPS:
            testing_data = convert_HDPS_data_record(testing_data, vocab_size)
        testing_dataset = SyntheticDatasetCollapsedLR(testing_data)

        if HDPS:
            PSLR = PropensityScoreLogisticRegression(1, vocab_size*3)
        else:
            PSLR = PropensityScoreLogisticRegression(1, vocab_size)

        model = model_dir + "PSLR_fold{}_model.pt".format(n)
        PSLR.load_state_dict(torch.load(model))
        
        ps_list, t_prob_list = get_propensity_score_PSLR(testing_dataset, PSLR)
        fig, (ax1, ax2) = plt.subplots(1, 2)
        sns.histplot(ps_list, kde=True, ax=ax1)
        ax1.set_xlim(0,1)
        ax1.set_xlabel("estimated propensity score")
        sns.histplot(t_prob_list, kde=True, ax=ax2)
        ax2.set_xlim(0,1)
        ax2.set_xlabel("true propensity score")
        plt.show()

def tune_batch_size_PSLR(data_dir, file_name, saving_dir, batch_size_list, nfold, vocab_size, training_epoch, 
                         learning_rate, early_stop, HDPS):
    
    tuning_result_dict = dict()
    data_nfold = np.array(load_nfold_data(data_dir, file_name, nfold)) # list of single folded split
    training_data = merge_folds(data_nfold[:-2])
    validation_data = data_nfold[-1]

    if HDPS:
        training_data = convert_HDPS_data_record(training_data, vocab_size)
        validation_data = convert_HDPS_data_record(validation_data, vocab_size)
    
    for batch_size in batch_size_list:
        
        print("building and initializing model...")
        training_dataset = SyntheticDatasetCollapsedBinary(training_data)
        validation_dataset = SyntheticDatasetCollapsedBinary(validation_data)

        if HDPS:
            best_model, result_dict = train_PSLR(training_dataset, validation_dataset, (vocab_size-1)*3+1, 
                                                 training_epoch, batch_size, learning_rate, early_stop=early_stop)
        else:
            best_model, result_dict = train_PSLR(training_dataset, validation_dataset, vocab_size, 
                                                 training_epoch, batch_size, learning_rate, early_stop=early_stop)
            
        tuning_result_dict[batch_size] = {"training_loss_per_epoch" : copy.deepcopy(result_dict["training_loss_per_epoch"]),
                                          "validation_loss_per_epoch" : copy.deepcopy(result_dict["validation_loss_per_epoch"])}
    
    print("saving result...")

    if HDPS:
        saving = saving_dir + "PSLR-HDPS_batch_size_tuning.pkl"
    else:
        saving = saving_dir + "PSLR_batch_size_tuning.pkl"

    save_data(saving, tuning_result_dict)
