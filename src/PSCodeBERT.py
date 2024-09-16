import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
import torchmetrics
import pickle
import os
import math
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from functools import partial
from src.SyntheticDataGeneration import *
from src.utils import *
import copy

def _get_activation_fn(activation):
    if activation == "relu":
        return nn.functional.relu
    elif activation == "gelu":
        return nn.functional.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        
    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = torch.nn.functional.relu
        super(TransformerEncoderLayer, self).__setstate__(state)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src2, weights = self.self_attn(src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src, weights
    
def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class TransformerEncoder(torch.nn.Module):
    __constants__ = ['norm']
    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, mask=None, src_key_padding_mask=None):
        output = src
        weights = []
        for mod in self.layers:
            output, weight = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
            weights.append(weight)

        if self.norm is not None:
            output = self.norm(output)
        return output, weights
    
class SegmentSinusoidalPositionalEncoding(nn.Module):

    def __init__(self, d_model: int, max_len: int):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len+1, 1, d_model) # idx 0 is for padding
        pe[1:, 0, 0::2] = torch.sin(position * div_term)
        pe[1:, 0, 1::2] = torch.cos(position * div_term)
        pe = pe.reshape((max_len+1, d_model))
        self.register_buffer('pe', pe)

    def forward(self, x, segment):
        """
        Arguments:
            x: Tensor, batch_size*max_seq_len*embedding_dim
        """
        return x + torch.nn.functional.embedding(segment, self.pe, padding_idx=0)

class PropensityScoreCodeBERT(torch.nn.Module):

    # BERT to predict propensity scores
    # each token corresponds to one code

    def __init__(self, embedding_dim, model_dim, vocab_size, num_layers, num_heads):
        super(PropensityScoreCodeBERT, self).__init__()

        self.model_dim = model_dim
        self.Embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.pos_encoder = SegmentSinusoidalPositionalEncoding(model_dim, 1000) # max_len manually set to 1000
        Encoder_Layer = TransformerEncoderLayer(d_model=model_dim, nhead=num_heads)
        self.Encoder = TransformerEncoder(Encoder_Layer, num_layers=num_layers)
        self.ps_prediction_fc = nn.Linear(model_dim, 1)
        self.token_prediction_fc = nn.Linear(model_dim, vocab_size)

    def forward(self, x, segment, mask):
        x = self.Embedding(x) * math.sqrt(self.model_dim) # batch_size * max_seq_len * embedding_dim
        x = self.pos_encoder(x, segment) # batch_size * max_seq_len * embedding_dim
        x = x.permute(1,0,2) # max_seq_len * batch_size * embedding_dim
        x, w = self.Encoder(x, src_key_padding_mask=mask) # x: max_seq_len * batch_size * embedding_dim 
        # w: list of batch_size * max_seq_len * max_seq_len
        x = x.permute(1,0,2) # x: batch_size * max_seq_len * embedding_dim 

        return self.token_prediction_fc(x), self.ps_prediction_fc(x), w
    
def pad_collate_codebert_masking(batch, CLS_idx=1, SEP_idx=2, MASK_idx=3, masking_prob=0.1):

    # collate_fn for training with MLM and prediction loss
    
    batch_size = len(batch)

    data, label, t_prob, outcome, data_len, keys = list(zip(*batch))
    
    data_flatten = []
    masked_data_flatten = []
    segment_list = []

    for rec in data:
        
        visit_flatten = [CLS_idx]
        segment_idx = [1]

        for i, v in enumerate(rec):
            v = np.array(v)
            visit_flatten.extend(v)
            visit_flatten.append(SEP_idx)
            segment_idx.extend([i+2]*(len(v)+1))
            
        visit_flatten = np.array(visit_flatten)
        masked_visit_flatten = copy.deepcopy(visit_flatten)
        masking_candidate_idx = np.argwhere(visit_flatten != SEP_idx).flatten()[1:] # SEP, CLS token is excluded from masking
        masking_num = int(np.max((np.floor(len(masking_candidate_idx)*masking_prob),1))) # minimum number of mask = 1
        masking_idx = np.random.choice(masking_candidate_idx, masking_num, replace=False)
        masked_visit_flatten[masking_idx] = MASK_idx
        
        segment_list.append(torch.Tensor(segment_idx[:-1])) # the last SEP token is excluded
        data_flatten.append(torch.Tensor(visit_flatten[:-1])) # the last SEP token is excluded
        masked_data_flatten.append(torch.Tensor(masked_visit_flatten[:-1])) # the last SEP token is excluded
        
    padded_data_flatten = torch.nn.utils.rnn.pad_sequence(data_flatten, batch_first=True, padding_value=0)
    padded_segment_list = torch.nn.utils.rnn.pad_sequence(segment_list, batch_first=True, padding_value=0)
    padded_masked_data_flatten = torch.nn.utils.rnn.pad_sequence(masked_data_flatten, batch_first=True, padding_value=0)
    mask = (padded_data_flatten == 0)

    return padded_data_flatten.long(), padded_masked_data_flatten.long(), padded_segment_list.long(), torch.Tensor(label), torch.Tensor(t_prob), torch.Tensor(outcome), mask

def pad_collate_codebert(batch, CLS_idx=1):

    # collate_fn for training with prediction loss 
    
    batch_size = len(batch)

    data, label, t_prob, outcome, data_len, keys = list(zip(*batch))
    
    data_flatten = []
    segment_list = []
    for rec in data:
        
        visit_flatten = [CLS_idx]
        segment_idx = [1]

        for i, v in enumerate(rec):
            v = np.array(v)
            visit_flatten.extend(v)
            segment_idx.extend([i+2]*(len(v)))
            
        visit_flatten = np.array(visit_flatten)
        
        segment_list.append(torch.Tensor(segment_idx)) 
        data_flatten.append(torch.Tensor(visit_flatten)) 
        
    padded_data_flatten = torch.nn.utils.rnn.pad_sequence(data_flatten, batch_first=True, padding_value=0)
    padded_segment_list = torch.nn.utils.rnn.pad_sequence(segment_list, batch_first=True, padding_value=0)
    mask = (padded_data_flatten == 0)
    
    return padded_data_flatten.long(), padded_data_flatten.long(), padded_segment_list.long(), torch.Tensor(label), torch.Tensor(t_prob), torch.Tensor(outcome), mask

def train_PSCodeBERT(training_dataset, validation_dataset,
                 embedding_dim, model_dim, vocab_size, num_layers, num_heads, 
                 training_epoch, batch_size, learning_rate, extra_token_idx, masking_prob=None, early_stop=True):
    
    CLS_idx, SEP_idx, MASK_idx = extra_token_idx
    
    print("loading data...")
    if masking_prob:
        # training with MLM and prediction loss
        training_dataloader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True, 
                                        collate_fn=partial(pad_collate_codebert_masking, CLS_idx=CLS_idx,SEP_idx=SEP_idx,MASK_idx=MASK_idx, masking_prob=masking_prob), 
                                        drop_last=True)
        validation_dataloader = DataLoader(validation_dataset, batch_size=10, shuffle=True, 
                                        collate_fn=partial(pad_collate_codebert_masking, CLS_idx=CLS_idx,SEP_idx=SEP_idx,MASK_idx=MASK_idx, masking_prob=masking_prob), 
                                        drop_last=True)
    else:
        training_dataloader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True, 
                                        collate_fn=partial(pad_collate_codebert, CLS_idx=CLS_idx), 
                                        drop_last=True)
        validation_dataloader = DataLoader(validation_dataset, batch_size=10, shuffle=True, 
                                        collate_fn=partial(pad_collate_codebert, CLS_idx=CLS_idx), 
                                        drop_last=True)
    print("loading data done...")
 
    print("building and initializing model...")
    PSBERT = PropensityScoreCodeBERT(embedding_dim, model_dim, vocab_size, num_layers, num_heads) 
    Sigmoid = torch.nn.Sigmoid()
    BCE_loss_fn = nn.BCELoss() # for prediction loss, this needs sigmoid
    CE_loss_fn = nn.CrossEntropyLoss(ignore_index=-1) # for MLM loss, this accepts logits
    Optimizer = torch.optim.Adam(PSBERT.parameters(), lr=learning_rate)
    print("building and initializing model done...") 
    
    print("calculating initial validation loss...")
    loss_sum = 0.
    ps_prediction_loss_sum = 0.
    mlm_loss_sum = 0.
    num_total_batch = len(validation_dataloader)
    PSBERT.eval()
    for x_batch, masked_x_batch, segment_batch, label_batch, t_prob_batch, outcome_batch, mask_batch in tqdm(validation_dataloader):

        with torch.no_grad():
            token_prediction, ps_prediction, _ = PSBERT(masked_x_batch, segment_batch, mask_batch) 
            loss_batch = 0

            ps_prediction = Sigmoid(ps_prediction) # batch_size * max_record_len * 1
            ps_prediction = ps_prediction[:,0,:] # the first token is CLS token batch_size * 1
            ps_prediction = ps_prediction.flatten()
            ps_prediction_loss = BCE_loss_fn(ps_prediction, label_batch)
            loss_batch += ps_prediction_loss
            ps_prediction_loss_sum += ps_prediction_loss.item()

            if masking_prob:
                mlm_label_batch = x_batch.clone()
                mlm_label_batch[masked_x_batch != MASK_idx] = -1 # non-masked tokens are replaced with -1 to be ignored 
                mlm_loss = CE_loss_fn(token_prediction.view(-1,vocab_size), mlm_label_batch.view(-1))
                loss_batch += mlm_loss
                mlm_loss_sum += mlm_loss.item()

            loss_sum += loss_batch.item()

    print("initial validation ps prediction loss: {}".format(ps_prediction_loss_sum/num_total_batch))
    print("initial validation mlm loss: {}".format(mlm_loss_sum/num_total_batch))
    print("initial validation loss: {}".format(loss_sum/num_total_batch))

    print("training starts...")
    training_loss_per_epoch = []
    training_ps_prediction_loss_per_epoch = []
    training_mlm_loss_per_epoch = []

    validation_loss_per_epoch = []
    validation_ps_prediction_loss_per_epoch = []
    validation_mlm_loss_per_epoch = []

    best_loss = np.inf
    best_model = None
    best_epoch = 0

    for e in range(training_epoch):

        if early_stop:
            if (e+1-best_epoch) >= 5:
                # if the loss did not decrease for 5 epoch in a row, stop training
                break

        print("epoch {}".format(e+1))
        loss_sum = 0.
        ps_prediction_loss_sum = 0.
        mlm_loss_sum = 0.
        num_total_batch = len(training_dataloader)

        PSBERT.train()
        for x_batch, masked_x_batch, segment_batch, label_batch, t_prob_batch, outcome_batch, mask_batch in tqdm(training_dataloader):
            
            Optimizer.zero_grad()
            token_prediction, ps_prediction, _ = PSBERT(masked_x_batch, segment_batch, mask_batch) 
            loss_batch = 0
                        
            ps_prediction = Sigmoid(ps_prediction) # batch_size * max_record_len * 1
            ps_prediction = ps_prediction[:,0,:] # the first token is CLS token batch_size * 1
            ps_prediction = ps_prediction.flatten()
            ps_prediction_loss = BCE_loss_fn(ps_prediction, label_batch)
            loss_batch += ps_prediction_loss
            ps_prediction_loss_sum += ps_prediction_loss.item()

            if masking_prob:
                mlm_label_batch = x_batch.clone()
                mlm_label_batch[masked_x_batch != MASK_idx] = -1 # non-masked tokens are replaced with -1 to be ignored 
                mlm_loss = CE_loss_fn(token_prediction.view(-1,vocab_size), mlm_label_batch.view(-1))
                loss_batch += mlm_loss
                mlm_loss_sum += mlm_loss.item()

            loss_batch.backward()
            Optimizer.step()
            loss_sum += loss_batch.item()

        training_loss = loss_sum/num_total_batch
        training_ps_prediction_loss = ps_prediction_loss_sum/num_total_batch
        training_mlm_loss = mlm_loss_sum/num_total_batch
        training_loss_per_epoch.append(training_loss)
        training_ps_prediction_loss_per_epoch.append(training_ps_prediction_loss)
        training_mlm_loss_per_epoch.append(training_mlm_loss)

        print("calculating validation loss...")
        loss_sum = 0.
        ps_prediction_loss_sum = 0.
        mlm_loss_sum = 0.
        num_total_batch = len(validation_dataloader)
        
        PSBERT.eval()
        for x_batch, masked_x_batch, segment_batch, label_batch, t_prob_batch, outcome_batch, mask_batch in tqdm(validation_dataloader):

            with torch.no_grad():
                token_prediction, ps_prediction, _ = PSBERT(masked_x_batch, segment_batch, mask_batch) 
                loss_batch = 0

                ps_prediction = Sigmoid(ps_prediction) # batch_size * max_record_len * 1
                ps_prediction = ps_prediction[:,0,:] # the first token is CLS token batch_size * 1
                ps_prediction = ps_prediction.flatten()
                ps_prediction_loss = BCE_loss_fn(ps_prediction, label_batch)
                loss_batch += ps_prediction_loss
                ps_prediction_loss_sum += ps_prediction_loss.item()

                if masking_prob:
                    mlm_label_batch = x_batch.clone()
                    mlm_label_batch[masked_x_batch != MASK_idx] = -1 # non-masked tokens are replaced with -1 to be ignored 
                    mlm_loss = CE_loss_fn(token_prediction.view(-1,vocab_size), mlm_label_batch.view(-1))
                    loss_batch += mlm_loss
                    mlm_loss_sum += mlm_loss.item()

                loss_sum += loss_batch.item()
        
        validation_loss = loss_sum/num_total_batch
        validation_ps_prediction_loss = ps_prediction_loss_sum/num_total_batch
        validation_mlm_loss = mlm_loss_sum/num_total_batch
        validation_loss_per_epoch.append(validation_loss)
        validation_ps_prediction_loss_per_epoch.append(validation_ps_prediction_loss)
        validation_mlm_loss_per_epoch.append(validation_mlm_loss)
        
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_epoch = e+1
            best_model = PSBERT.state_dict()

        print("training ps prediction loss: {}".format(training_ps_prediction_loss))
        print("training mlm loss: {}".format(training_mlm_loss))
        print("training loss: {}".format(training_loss))
        print("--------------------------------------")
        print("validation ps prediction loss: {}".format(validation_ps_prediction_loss))
        print("validation mlm loss: {}".format(validation_mlm_loss))
        print("validation loss: {}".format(validation_loss))
            
    print("saving results...")
    result_dict = {"training_loss_per_epoch" : training_loss_per_epoch, "validation_loss_per_epoch" : validation_loss_per_epoch,
                   "training_ps_prediction_loss_per_epoch" : training_ps_prediction_loss_per_epoch,
                   "training_mlm_loss_per_epoch" : training_mlm_loss_per_epoch,
                   "validation_ps_prediction_loss_per_epoch" : validation_ps_prediction_loss_per_epoch, 
                   "validation_mlm_loss_per_epoch" : validation_mlm_loss_per_epoch}

    PSBERT.load_state_dict(best_model)

    return PSBERT, result_dict

def evaluate_PSCodeBERT(testing_dataset, PSBERT_model, extra_token_idx, masking_prob=None):

    CLS_idx, SEP_idx, MASK_idx = extra_token_idx

    if masking_prob:
        testing_dataloader = DataLoader(testing_dataset, batch_size=10, shuffle=True, 
                                        collate_fn=partial(pad_collate_codebert_masking, CLS_idx=CLS_idx,SEP_idx=SEP_idx,MASK_idx=MASK_idx, masking_prob=masking_prob), 
                                        drop_last=True)
    else:
        testing_dataloader = DataLoader(testing_dataset, batch_size=10, shuffle=True, 
                                        collate_fn=partial(pad_collate_codebert, CLS_idx=CLS_idx), 
                                        drop_last=True)

    Sigmoid = torch.nn.Sigmoid()
    MAE = torchmetrics.MeanAbsoluteError()
    weighted_MAE = torchmetrics.MeanAbsoluteError()
    
    print("evaluation starts...")
    y_z1_sum = 0.
    z1_sum = 0.
    y_z0_sum = 0.
    z0_sum = 0.
        
    PSBERT_model.eval()
    for x_batch, masked_x_batch, segment_batch, label_batch, t_prob_batch, outcome_batch, mask_batch in tqdm(testing_dataloader):

        with torch.no_grad():           
            
            token_prediction, ps_prediction, _ = PSBERT_model(x_batch, segment_batch, mask_batch) # use non-masked batch for evaluation
            ps_prediction = Sigmoid(ps_prediction) # batch_size * max_record_len * 1
            ps_prediction = ps_prediction[:,0,:] # the first token is CLS token batch_size * 1
            ps_prediction = ps_prediction.flatten()
            MAE.update(ps_prediction, t_prob_batch)
            weighted_MAE.update(ps_prediction * t_prob_batch, t_prob_batch * t_prob_batch)

            # compute IPW-ATE
            y_z1, z1, y_z0, z0 = compute_IPW_ATE_partialsum(label_batch, outcome_batch, ps_prediction)
            y_z1_sum += y_z1.item()
            z1_sum += z1.item()
            y_z0_sum += y_z0.item()
            z0_sum += z0.item()
    
    ipw_ate = (y_z1_sum-y_z0_sum) / (z1_sum+z0_sum)
    print("evaluation done...")
    return MAE.compute().item(), weighted_MAE.compute().item(), ipw_ate

def evaluate_PSCodeBERT_clip(testing_dataset, PSBERT_model, clip_value, extra_token_idx, masking_prob=None):

    CLS_idx, SEP_idx, MASK_idx = extra_token_idx

    if masking_prob:
        testing_dataloader = DataLoader(testing_dataset, batch_size=10, shuffle=True, 
                                        collate_fn=partial(pad_collate_codebert_masking, CLS_idx=CLS_idx,SEP_idx=SEP_idx,MASK_idx=MASK_idx, masking_prob=masking_prob), 
                                        drop_last=True)
    else:
        testing_dataloader = DataLoader(testing_dataset, batch_size=10, shuffle=True, 
                                        collate_fn=partial(pad_collate_codebert, CLS_idx=CLS_idx), 
                                        drop_last=True)

    Sigmoid = torch.nn.Sigmoid()
    MAE = torchmetrics.MeanAbsoluteError()
    
    print("evaluation starts...") # MAE, IPW-ATE
    y_z1_sum = 0.
    z1_sum = 0.
    y_z0_sum = 0.
    z0_sum = 0.
        
    PSBERT_model.eval()
    for x_batch, masked_x_batch, segment_batch, label_batch, t_prob_batch, outcome_batch, mask_batch in tqdm(testing_dataloader):

        with torch.no_grad():           
            
            token_prediction, ps_prediction, _ = PSBERT_model(x_batch, segment_batch, mask_batch) 
            ps_prediction = Sigmoid(ps_prediction) # batch_size * max_record_len * 1
            ps_prediction = ps_prediction[:,0,:] # the first token is CLS token batch_size * 1
            ps_prediction = ps_prediction.flatten()
            ps_prediction = torch.clamp(ps_prediction, min=clip_value[0], max=clip_value[1])
            MAE.update(ps_prediction, t_prob_batch)

            # compute IPW-ATE
            y_z1, z1, y_z0, z0 = compute_IPW_ATE_partialsum(label_batch, outcome_batch, ps_prediction)
            y_z1_sum += y_z1.item()
            z1_sum += z1.item()
            y_z0_sum += y_z0.item()
            z0_sum += z0.item()
    
    ipw_ate = (y_z1_sum-y_z0_sum) / (z1_sum+z0_sum)
    print("evaluation done...")
    return MAE.compute().item(), ipw_ate

def evaluate_PSCodeBERT_trimming(testing_dataset, PSBERT_model, trimming_range, extra_token_idx, masking_prob=None):

    CLS_idx, SEP_idx, MASK_idx = extra_token_idx
    
    if masking_prob:
        testing_dataloader = DataLoader(testing_dataset, batch_size=10, shuffle=True, 
                                        collate_fn=partial(pad_collate_codebert_masking, CLS_idx=CLS_idx,SEP_idx=SEP_idx,MASK_idx=MASK_idx, masking_prob=masking_prob), 
                                        drop_last=True)
    else:
        testing_dataloader = DataLoader(testing_dataset, batch_size=10, shuffle=True, 
                                        collate_fn=partial(pad_collate_codebert, CLS_idx=CLS_idx), 
                                        drop_last=True)

    Sigmoid = torch.nn.Sigmoid()
    MAE = torchmetrics.MeanAbsoluteError()
    
    print("evaluation starts...") # MAE, IPW-ATE
    y_z1_sum = 0.
    z1_sum = 0.
    y_z0_sum = 0.
    z0_sum = 0.
        
    PSBERT_model.eval()
    for x_batch, masked_x_batch, segment_batch, label_batch, t_prob_batch, outcome_batch, mask_batch in tqdm(testing_dataloader):

        with torch.no_grad():           
            
            token_prediction, ps_prediction, _ = PSBERT_model(x_batch, segment_batch, mask_batch) 
            ps_prediction = Sigmoid(ps_prediction) # batch_size * max_record_len * 1
            ps_prediction = ps_prediction[:,0,:] # the first token is CLS token batch_size * 1
            ps_prediction = ps_prediction.flatten()

            trimming_ps_prediction = ps_prediction[(ps_prediction >= trimming_range[0]) & (ps_prediction <= trimming_range[1])]
            t_prob_batch = t_prob_batch[(ps_prediction >= trimming_range[0]) & (ps_prediction <= trimming_range[1])]
            label_batch = label_batch[(ps_prediction >= trimming_range[0]) & (ps_prediction <= trimming_range[1])]
            outcome_batch = outcome_batch[(ps_prediction >= trimming_range[0]) & (ps_prediction <= trimming_range[1])]
            MAE.update(trimming_ps_prediction, t_prob_batch)

            # compute IPW-ATE
            y_z1, z1, y_z0, z0 = compute_IPW_ATE_partialsum(label_batch, outcome_batch, trimming_ps_prediction)
            y_z1_sum += y_z1.item()
            z1_sum += z1.item()
            y_z0_sum += y_z0.item()
            z0_sum += z0.item()
    
    ipw_ate = (y_z1_sum-y_z0_sum) / (z1_sum+z0_sum)
    print("evaluation done...")
    return MAE.compute().item(), ipw_ate

def nfold_train_PSCodeBERT(data_dir, file_name, saving_dir, nfold, embedding_dim, model_dim, vocab_size, num_layers, num_heads, 
                     training_epoch, batch_size, learning_rate, extra_token_idx, masking_prob=None, early_stop=True):

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

        best_model, result_dict = train_PSCodeBERT(training_dataset, validation_dataset, 
                                               embedding_dim, model_dim, vocab_size, num_layers, num_heads,
                                             training_epoch, batch_size, learning_rate, extra_token_idx, masking_prob=masking_prob, 
                                             early_stop=early_stop)
        ps_mae, weighted_ps_mae, ipw_ate = evaluate_PSCodeBERT(testing_dataset, best_model, extra_token_idx, masking_prob=masking_prob)
        print("IPW-ATE : {}".format(ipw_ate))
        print("PS-MAE : {}".format(ps_mae))
        print("weighted PS-MAE : {}".format(weighted_ps_mae))

        nfold_result_dict[n] = copy.deepcopy(result_dict)

        torch.save(best_model.state_dict(), saving_dir+"PSCodeBERT_fold{}_model.pt".format(n))

    save_data(saving_dir+"PSCodeBERT_result_dict.pkl", nfold_result_dict)

def nfold_evaluate_PSCodeBERT(data_dir, file_name, model_dir, nfold, embedding_dim, model_dim, vocab_size, num_layers, num_heads, 
                          extra_token_idx, trimming_range, clip_range, masking_prob=None):

    nfold_evaluation_dict = dict()
    data_nfold = np.array(load_nfold_data(data_dir, file_name, nfold)) # list of single folded split
    nfold_idx = np.tile(np.arange(nfold),2)

    for n in range(nfold):

        testing_idx = nfold_idx[n+1]
        testing_data = data_nfold[testing_idx]
        testing_dataset = SyntheticDataset(testing_data)

        PSBERT = PropensityScoreCodeBERT(embedding_dim, model_dim, vocab_size, num_layers, num_heads) 
        model = model_dir + "PSCodeBERT_fold{}_model.pt".format(n)
        PSBERT.load_state_dict(torch.load(model))

        ps_mae, weighted_ps_mae, ipw_ate = evaluate_PSCodeBERT(testing_dataset, PSBERT, extra_token_idx, masking_prob=masking_prob)
        print("IPW-ATE : {}".format(ipw_ate))
        print("PS-MAE : {}".format(ps_mae))
        print("weighted PS-MAE : {}".format(weighted_ps_mae))

        if trimming_range:
            trimming_ps_mae, trimming_ipw_ate = evaluate_PSCodeBERT_trimming(testing_dataset, PSBERT, trimming_range, extra_token_idx, masking_prob=masking_prob)
            print("IPW-ATE with trimming: {}".format(trimming_ipw_ate))
            print("PS-MAE with trimming: {}".format(trimming_ps_mae))

        if clip_range:
            clip_ps_mae, clip_ipw_ate = evaluate_PSCodeBERT_clip(testing_dataset, PSBERT, clip_range, extra_token_idx, masking_prob=masking_prob)
            print("IPW-ATE with clip: {}".format(clip_ipw_ate))
            print("PS-MAE with clip: {}".format(clip_ps_mae))

        nfold_evaluation_dict[n] = {"PS_MAE" : ps_mae, "IPW_ATE" : ipw_ate, "weighted PS_MAE" : weighted_ps_mae}
        
        if trimming_range:
            nfold_evaluation_dict[n].update({"trimming_PS_MAE" : trimming_ps_mae, "trimming_IPW_ATE" : trimming_ipw_ate})

        if clip_range:
            nfold_evaluation_dict[n].update({"clip_PS_MAE" : clip_ps_mae, "clip_IPW_ATE" : clip_ipw_ate})

    return nfold_evaluation_dict

def get_propensity_score_PSCodeBERT(testing_dataset, PSBERT_model, extra_token_idx, masking_prob=None):

    CLS_idx, SEP_idx, MASK_idx = extra_token_idx
    
    if masking_prob:
        testing_dataloader = DataLoader(testing_dataset, batch_size=10, shuffle=True, 
                                        collate_fn=partial(pad_collate_codebert_masking, CLS_idx=CLS_idx,SEP_idx=SEP_idx,MASK_idx=MASK_idx, masking_prob=masking_prob), 
                                        drop_last=True)
    else:
        testing_dataloader = DataLoader(testing_dataset, batch_size=10, shuffle=True, 
                                        collate_fn=partial(pad_collate_codebert, CLS_idx=CLS_idx), 
                                        drop_last=True)

    Sigmoid = torch.nn.Sigmoid()
    ps_list = []
    t_prob_list = []

    PSBERT_model.eval()
    for x_batch, masked_x_batch, segment_batch, label_batch, t_prob_batch, outcome_batch, mask_batch in tqdm(testing_dataloader):

        with torch.no_grad():           
            
            token_prediction, ps_prediction, _ = PSBERT_model(x_batch, segment_batch, mask_batch) 
            ps_prediction = Sigmoid(ps_prediction) # batch_size * max_record_len * 1
            ps_prediction = ps_prediction[:,0,:] # the first token is CLS token batch_size * 1
            ps_prediction = ps_prediction.flatten()
            
            ps_list.extend(ps_prediction.tolist())
            t_prob_list.extend(t_prob_batch.tolist())
            
    return ps_list, t_prob_list

def nfold_plot_propensity_score_PSCodeBERT(data_dir, file_name, model_dir, nfold, embedding_dim, model_dim, 
                                vocab_size, num_layers, num_heads, extra_token_idx, masking_prob=None):

    data_nfold = np.array(load_nfold_data(data_dir, file_name, nfold)) # list of single folded split
    nfold_idx = np.tile(np.arange(nfold),2)
    
    for n in range(nfold):

        testing_idx = nfold_idx[n+1]
        testing_data = data_nfold[testing_idx]
        testing_dataset = SyntheticDataset(testing_data)

        PSBERT = PropensityScoreCodeBERT(embedding_dim, model_dim, vocab_size, num_layers, num_heads) 
        model = model_dir + "PSCodeBERT_fold{}_model.pt".format(n)
        PSBERT.load_state_dict(torch.load(model))
        
        ps_list, t_prob_list = get_propensity_score_PSCodeBERT(testing_dataset, PSBERT, extra_token_idx, masking_prob=masking_prob)
        fig, (ax1, ax2) = plt.subplots(1, 2)
        sns.histplot(ps_list, kde=True, ax=ax1)
        ax1.set_xlim(0,1)
        ax1.set_xlabel("estimated propensity score")
        sns.histplot(t_prob_list, kde=True, ax=ax2)
        ax2.set_xlim(0,1)
        ax2.set_xlabel("true propensity score")
        plt.show()

def get_average_attention_PSCodeBERT(testing_dataset, PSBERT_model, confounding_var, extra_token_idx, masking_prob):
    
    CLS_idx, SEP_idx, MASK_idx = extra_token_idx

    if masking_prob:
        testing_dataloader = DataLoader(testing_dataset, batch_size=10, shuffle=True, 
                                        collate_fn=partial(pad_collate_codebert_masking, CLS_idx=CLS_idx,SEP_idx=SEP_idx,MASK_idx=MASK_idx, masking_prob=masking_prob), 
                                        drop_last=True)
    else:
        testing_dataloader = DataLoader(testing_dataset, batch_size=10, shuffle=True, 
                                        collate_fn=partial(pad_collate_codebert, CLS_idx=CLS_idx), 
                                        drop_last=True)

    Sigmoid = torch.nn.Sigmoid()
    
    PSBERT_model.eval()
    for x_batch, masked_x_batch, segment_batch, label_batch, t_prob_batch, outcome_batch, mask_batch in tqdm(testing_dataloader):

        with torch.no_grad():           
            token_prediction, ps_prediction, attention_weights = PSBERT_model(x_batch, segment_batch, mask_batch)
            ps_prediction = Sigmoid(ps_prediction) # batch_size * max_record_len * 1
            ps_prediction = ps_prediction[:,0,:] # the first token is CLS token batch_size * 1
            ps_prediction = ps_prediction.flatten()
        
        # attention weight of the last layer corresponding to CLS token
        CLS_attention_weights = attention_weights[-1][:,0,:].detach()
        x_batch_exclude_CLS = x_batch[:,1:]
        CLS_attention_weights_exclude_itself = attention_weights[-1][:,0,1:].detach()
        
        target_attention_weight_mean = 0.
        rest_attention_weight_mean = 0.
        CLS_attention_weight_mean = 0.
        
        for c_var in confounding_var:
            # confounding_var could be multiple
            target_attention_weights = CLS_attention_weights[x_batch == c_var]
            rest_attention_weights = CLS_attention_weights_exclude_itself[x_batch_exclude_CLS != c_var]
            
            target_attention_weight_mean += target_attention_weights.mean().item()
            rest_attention_weight_mean += rest_attention_weights.mean().item()
            CLS_attention_weight_mean += CLS_attention_weights[:,0].mean().item()
            
        target_attention_weight_mean = target_attention_weight_mean/len(confounding_var)
        rest_attention_weight_mean = rest_attention_weight_mean/len(confounding_var)
        CLS_attention_weight_mean = CLS_attention_weight_mean/len(confounding_var)
        
    return target_attention_weight_mean, rest_attention_weight_mean, CLS_attention_weight_mean

def nfold_average_attention_PSCodeBERT(data_dir, file_name, params_dir, model_dir, nfold, embedding_dim, model_dim, vocab_size, num_layers, 
                                       num_heads, extra_token_idx):
    
    # calculate average attention weight of the last layer corresponding to confounding variables across n-folds
    # output: 
    
    data_nfold = np.array(load_nfold_data(data_dir, file_name, nfold)) # list of single folded split
    nfold_idx = np.tile(np.arange(nfold),2)
    params_dict = load_data(params_dir)
    confounding_var = params_dict["confounding_var"] + params_dict["num_token_reserve"] 
    
    target_attention_weight_list = []
    rest_attention_weight_list = []
    CLS_attention_weight_list = []

    for n in range(nfold):

        testing_idx = nfold_idx[n+1]
        testing_data = data_nfold[testing_idx]
        testing_dataset = SyntheticDataset(testing_data)

        PSBERT = PropensityScoreCodeBERT(embedding_dim, model_dim, vocab_size, num_layers, num_heads) 
        model = model_dir + "PSCodeBERT_fold{}_model.pt".format(n)
        PSBERT.load_state_dict(torch.load(model))

        target_attention_weight_avg, rest_attention_weight_avg, CLS_attention_weight_avg = get_average_attention_PSCodeBERT(testing_dataset, PSBERT, confounding_var, extra_token_idx, masking_prob=None)
        
        target_attention_weight_list.append(target_attention_weight_avg)
        rest_attention_weight_list.append(rest_attention_weight_avg)
        CLS_attention_weight_list.append(CLS_attention_weight_avg)
    
    print("target_attention_weight_avg: {}".format(np.mean(target_attention_weight_list)))
    print("target_attention_weight 95% CI: {}".format(st.t.interval(0.95, len(target_attention_weight_list)-1, 
                                                    loc=np.mean(target_attention_weight_list), scale=st.sem(target_attention_weight_list))))
    print("target_attention_weight CI half width: {}".format(st.t.interval(0.95, len(target_attention_weight_list)-1, loc=np.mean(target_attention_weight_list), scale=st.sem(target_attention_weight_list))[0] - np.mean(target_attention_weight_list)))

    print("----------------------")
    print("rest_attention_weight_avg: {}".format(np.mean(rest_attention_weight_list)))
    print("rest_attention_weight 95% CI: {}".format(st.t.interval(0.95, len(rest_attention_weight_list)-1, 
                                                    loc=np.mean(rest_attention_weight_list), scale=st.sem(rest_attention_weight_list))))
    print("rest_attention_weight CI half width: {}".format(st.t.interval(0.95, len(rest_attention_weight_list)-1, loc=np.mean(rest_attention_weight_list), scale=st.sem(rest_attention_weight_list))[0] - np.mean(rest_attention_weight_list)))

    
    print("----------------------")
    print("CLS_attention_weight_avg: {}".format(np.mean(CLS_attention_weight_list)))
    print("CLS_attention_weight 95% CI: {}".format(st.t.interval(0.95, len(CLS_attention_weight_list)-1, 
                                                    loc=np.mean(CLS_attention_weight_list), scale=st.sem(CLS_attention_weight_list))))
    print("CLS_attention_weight CI half width: {}".format(st.t.interval(0.95, len(CLS_attention_weight_list)-1, loc=np.mean(CLS_attention_weight_list), scale=st.sem(CLS_attention_weight_list))[0] - np.mean(CLS_attention_weight_list)))

    
def get_confouding_var_idx(x_vector, confounding_var):
    
    confounding_var_idx = []
    
    # confounding_var could be list
    
    for c_var in confounding_var:
        confounding_var_idx.extend(torch.argwhere(x_vector == c_var).flatten().tolist())
        
    confounding_var_idx = list(np.sort(confounding_var_idx))
    
    return confounding_var_idx

def plot_attention_PSCodeBERT(data_dir, file_name, params_dir, model_dir, nfold, embedding_dim, model_dim, vocab_size, num_layers,
                             num_heads, num_example, fold_selection, extra_token_idx, masking_prob=None, save=None):
    # add saving option
    CLS_idx, SEP_idx, MASK_idx = extra_token_idx
    
    data_nfold = np.array(load_nfold_data(data_dir, file_name, nfold)) # list of single folded split
    nfold_idx = np.tile(np.arange(nfold),2)
    params_dict = load_data(params_dir)
    confounding_var = params_dict["confounding_var"] + params_dict["num_token_reserve"] 

    testing_idx = nfold_idx[fold_selection+1]
    testing_data = data_nfold[testing_idx]
    testing_dataset = SyntheticDataset(testing_data)
    
    if masking_prob:
        testing_dataloader = DataLoader(testing_dataset, batch_size=num_example, shuffle=True, 
                                        collate_fn=partial(pad_collate_codebert_masking, CLS_idx=CLS_idx,SEP_idx=SEP_idx,MASK_idx=MASK_idx, masking_prob=masking_prob), 
                                        drop_last=True)
    else:
        testing_dataloader = DataLoader(testing_dataset, batch_size=num_example, shuffle=True, 
                                        collate_fn=partial(pad_collate_codebert, CLS_idx=CLS_idx), 
                                        drop_last=True)
        
    PSBERT = PropensityScoreCodeBERT(embedding_dim, model_dim, vocab_size, num_layers, num_heads) 
    model = model_dir + "PSCodeBERT_fold{}_model.pt".format(fold_selection)
    PSBERT.load_state_dict(torch.load(model))
    Sigmoid = torch.nn.Sigmoid()
    
    PSBERT.eval()
    x_batch, masked_x_batch, segment_batch, label_batch, t_prob_batch, outcome_batch, mask_batch = next(iter(testing_dataloader))

    with torch.no_grad():           
        token_prediction, ps_prediction, attention_weights = PSBERT(x_batch, segment_batch, mask_batch)
        ps_prediction = Sigmoid(ps_prediction) # batch_size * max_record_len * 1
        ps_prediction = ps_prediction[:,0,:] # the first token is CLS token batch_size * 1
        ps_prediction = ps_prediction.flatten()
        
    len_batch = mask_batch.size(-1) - mask_batch.sum(axis=1)
    
    for i in range(num_example):
        attention_matrix = attention_weights[-1][i,:len_batch[i],:len_batch[i]]
        attention_matrix = attention_matrix.detach().numpy()
        confounding_var_idx = get_confouding_var_idx(x_batch[i], confounding_var)
        fig, ax = plt.subplots(figsize=(3.26*2/2, 3))  # set figure size
        heatmap = ax.pcolor(attention_matrix, cmap=plt.cm.Blues, alpha=0.9)
        
        ticks = [0]
        tick_labels = ["[CLS]"]
        
        ticks.extend(confounding_var_idx)
        ticks = np.array(ticks) + 0.5
        tick_labels.extend(["C"] * len(confounding_var_idx))
        
        ax.set_xticks(ticks, tick_labels)
        ax.set_yticks(ticks, tick_labels)
        
        if save != None:
            saving = save + "PSCodeBERT_attention_map_{}.pdf".format(i)
            fig.tight_layout()
            plt.savefig(saving, dpi=300, format="pdf")
        
    return attention_weights, ps_prediction, t_prob_batch
    

# need to revise
def tune_batch_size_PSBERT(data_dir, file_name, saving_dir, batch_size_list, nfold, embedding_dim, model_dim, vocab_size, 
                           num_layers, num_heads, training_epoch, learning_rate, early_stop):
    
    tuning_result_dict = dict()
    data_nfold = np.array(load_nfold_data(data_dir, file_name, nfold)) # list of single folded split
    training_data = merge_folds(data_nfold[:-2])
    validation_data = data_nfold[-1]
    
    for batch_size in batch_size_list:
        
        print("building and initializing model...")
        training_dataset = SyntheticDataset(training_data)
        validation_dataset = SyntheticDataset(validation_data)

        best_model, result_dict = train_PSBERT(training_dataset, validation_dataset, 
                                               embedding_dim, model_dim, vocab_size, num_layers, num_heads,
                                               training_epoch, batch_size, learning_rate, early_stop)

        tuning_result_dict[batch_size] = {"training_loss_per_epoch" : copy.deepcopy(result_dict["training_loss_per_epoch"]),
                                          "validation_loss_per_epoch" : copy.deepcopy(result_dict["validation_loss_per_epoch"])}
    
    print("saving result...")
    saving = saving_dir + "PSBERT_batch_size_tuning.pkl"
    save_data(saving, tuning_result_dict)