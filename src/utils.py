import torch
import numpy as np
import pickle
import scipy.stats as st
import torchmetrics

def load_data(data_dir):
    file = open(data_dir,'rb')
    
    return pickle.load(file)

def save_data(save_dir, data_dict):
    with open(save_dir, 'wb') as f:
        pickle.dump(data_dict, f)

def split_data_nfold(data_record, nfold):

    keys = list(data_record.keys())
    np.random.shuffle(keys)
    fold_size = int(len(data_record)/nfold)
    fold_data_list = []
    
    for n in range(nfold):
        fold_keys = keys[n*fold_size:(n+1)*fold_size]
        fold_data_record = dict()
        
        for k in fold_keys:
            fold_data_record[k] = data_record[k]
            
        fold_data_list.append(fold_data_record)

    return fold_data_list

def merge_folds(fold_list):
    
    merged_dict = dict()
    
    for fold in fold_list:
        merged_dict.update(fold)
        
    return merged_dict

def compute_IPW_ATE_partialsum(t_label, outcome, propensity_score):

    # compute IPW_ATE partial sum for a batch
    propensity_score = torch.clip(propensity_score, 0.01,0.99) # clipping propensity score for numerical stability

    w1 = 1 / propensity_score
    w0 = 1 / (1-propensity_score)

    y_z1 = (t_label*outcome*w1).sum()
    z1 = t_label.sum()
    y_z0 = ((1-t_label)*outcome*w0).sum()
    z0 = (1-t_label).sum()

    return y_z1, z1, y_z0, z0

def load_nfold_data(data_dir, file_name, nfold):

    nfold_list = []

    for n in range(nfold):
        print("loading {}-fold data".format(n+1))
        loading = data_dir + file_name + "_{}fold.pkl".format(n)
        nfold_list.append(load_data(loading))

    return nfold_list

def empty_visit_check(code_binary):
    
    sum_per_visit = np.sum(code_binary, 1)
    
    return np.sum(sum_per_visit == 0), sum_per_visit

def summarize_evaluation_dict(evaluation_dict, treatment_effect):
    
    ps_mae_list = []
    weighted_ps_mae_list = []
    confounded_ps_mae_list = []
    ipw_ate_list = []
    trimming_ipw_ate_list = []
    clip_ipw_ate_list = []    
    
    for fold, v in evaluation_dict.items():
        ps_mae_list.append(v["PS_MAE"])
        weighted_ps_mae_list.append(v["weighted PS_MAE"]) 
        ipw_ate_list.append(v["IPW_ATE"])
        trimming_ipw_ate_list.append(v["trimming_IPW_ATE"])
        clip_ipw_ate_list.append(v["clip_IPW_ATE"])
        
    print("PS_MAE mean: {}".format(np.mean(ps_mae_list)))
    print("PS_MAE 95% CI: {}".format(st.t.interval(0.95, len(ps_mae_list)-1, loc=np.mean(ps_mae_list), scale=st.sem(ps_mae_list))))
    print("PS_MAE CI half width: {}".format(st.t.interval(0.95, len(ps_mae_list)-1, loc=np.mean(ps_mae_list), scale=st.sem(ps_mae_list))[0] - np.mean(ps_mae_list)))
    print("weighted PS_MAE mean: {}".format(np.mean(weighted_ps_mae_list)))
    print("weighted PS_MAE 95% CI: {}".format(st.t.interval(0.95, len(weighted_ps_mae_list)-1, 
                                                            loc=np.mean(weighted_ps_mae_list), scale=st.sem(weighted_ps_mae_list))))
    print("weighted PS_MAE CI half width: {}".format(st.t.interval(0.95, len(weighted_ps_mae_list)-1, loc=np.mean(weighted_ps_mae_list), scale=st.sem(weighted_ps_mae_list))[0] - np.mean(weighted_ps_mae_list)))


    treatment_effect_tensor = torch.Tensor([treatment_effect])
    treatment_effect_tensor = treatment_effect_tensor.tile(len(evaluation_dict))
    MAE = torchmetrics.MeanAbsoluteError()

    mae_ipw_ate = MAE(torch.Tensor(ipw_ate_list), treatment_effect_tensor).item()
    trimming_mae_ipw_ate = MAE(torch.Tensor(trimming_ipw_ate_list), treatment_effect_tensor).item()
    clip_mae_ipw_ate = MAE(torch.Tensor(clip_ipw_ate_list), treatment_effect_tensor).item()
    print("----------------------")
    print("MAE_IPW_ATE: {}".format(mae_ipw_ate))
    print("IPW_ATE 95% CI: {}".format(st.t.interval(0.95, len(ipw_ate_list)-1, 
                                                    loc=np.mean(ipw_ate_list), scale=st.sem(ipw_ate_list))))
    print("IPW_ATE CI half width: {}".format(st.t.interval(0.95, len(ipw_ate_list)-1, loc=np.mean(ipw_ate_list), scale=st.sem(ipw_ate_list))[0] - np.mean(ipw_ate_list)))

    print("----------------------")
    print("trimming MAE_IPW_ATE: {}".format(trimming_mae_ipw_ate))
    print("trimming IPW_ATE 95% CI: {}".format(st.t.interval(0.95, len(trimming_ipw_ate_list)-1, 
                                                         loc=np.mean(trimming_ipw_ate_list), scale=st.sem(trimming_ipw_ate_list))))
    print("trimming IPW_ATE CI half width: {}".format(st.t.interval(0.95, len(trimming_ipw_ate_list)-1, loc=np.mean(trimming_ipw_ate_list), scale=st.sem(trimming_ipw_ate_list))[0] - np.mean(trimming_ipw_ate_list)))

    print("----------------------")
    print("clip MAE_IPW_ATE: {}".format(clip_mae_ipw_ate))
    print("clip IPW_ATE 95% CI: {}".format(st.t.interval(0.95, len(clip_ipw_ate_list)-1, 
                                                         loc=np.mean(clip_ipw_ate_list), scale=st.sem(clip_ipw_ate_list))))
    print("clip IPW_ATE CI half width: {}".format(st.t.interval(0.95, len(clip_ipw_ate_list)-1, loc=np.mean(clip_ipw_ate_list), scale=st.sem(clip_ipw_ate_list))[0] - np.mean(clip_ipw_ate_list)))