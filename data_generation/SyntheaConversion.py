import numpy as np
import pickle
import torch
from tqdm import tqdm
from src.utils import *
import copy
from sklearn.preprocessing import StandardScaler

def rebuild_code_dict(code2id, num_token_reserve):

    new_code2id = dict()
    new_id2code = dict()

    for code, code_id in code2id.items():

        new_code2id[code] = code_id + num_token_reserve
        new_id2code[code_id + num_token_reserve] = code
    
    return new_code2id, new_id2code

def calculate_dependency_distance(binary_code, temporal_confounding_var_idx, fixed_confounding_var_idx):
    
    temporal_confounding_var_code_occurrence = binary_code[:, temporal_confounding_var_idx]
    fixed_confounding_var_code_occurrence = binary_code[:, fixed_confounding_var_idx]
    
    temporal_confounding_occurrence_idx = temporal_confounding_var_code_occurrence.nonzero()[0][0]
    fixed_confounding_occurrence_idx = fixed_confounding_var_code_occurrence.nonzero()[0]
    
    dependency_distance = np.min(np.abs(fixed_confounding_occurrence_idx - temporal_confounding_occurrence_idx))
    
    return dependency_distance

def generate_dependency_distance_treatment_prob(dependency_distance, noise_var):

    noise = np.random.normal(0, noise_var)

    distance_effect = 2*np.log10(10/((dependency_distance**2.5)))
    treatment_prob = torch.sigmoid(torch.tensor(distance_effect)) + noise
    treatment_prob = torch.clip(treatment_prob, 0.01, 0.99)
    
    return treatment_prob

def generate_dependency_distance_confounding_outcome(treatment_effect, treatment_assignment, 
                                                     alpha, bias, dependency_distance, noise_var):
    
    noise = np.random.normal(0, noise_var)

    confounding_outcome = alpha * (1/(dependency_distance))

    return bias + treatment_effect*treatment_assignment + confounding_outcome + noise

class PatientRecordSynthea:
    
    def __init__(self, data_dict, code2id, id2code, code2desc, params):

        self.data_dict = data_dict
        self.code2id = code2id # code2id should not include reserved tokens
        self.id2code = id2code
        self.code2desc = code2desc
        self.params = params
        self.patient_record = dict()

    def set_confounding_vars(self, temporal_confounding_var_idx, fixed_confounding_var_idx):

        print("setting temporal confounding variable idx {},{}".format(temporal_confounding_var_idx, self.code2desc[self.id2code[temporal_confounding_var_idx]]))
        print("setting fixed confounding variable idx {},{}".format(fixed_confounding_var_idx, self.code2desc[self.id2code[fixed_confounding_var_idx]]))
        self.temporal_confounding_var_idx = temporal_confounding_var_idx
        self.fixed_confounding_var_idx = fixed_confounding_var_idx

    def rebuild_code_dict(self, saving_dir):

        # rebuild code2id, id2code with reserved tokens
        self.code2id_PSMLP, self.id2code_PSMLP = rebuild_code_dict(self.code2id, 1)
        save_data(saving_dir + "code2id_PSMLP.pkl", self.code2id_PSMLP)
        save_data(saving_dir + "id2code_PSMLP.pkl", self.id2code_PSMLP)

        self.code2id_sequential, self.id2code_sequential = rebuild_code_dict(self.code2id, self.params["num_token_reserve"])
        save_data(saving_dir + "code2id_sequential.pkl", self.code2id_sequential)
        save_data(saving_dir + "id2code_sequential.pkl", self.id2code_sequential)

    def save_record(self, saving_dir):
        saving = saving_dir + "synthea_sinusitis_patient_record.pkl"
        save_data(saving, self.patient_record)

    def split_record_nfold(self, nfold):
        
        fold_record_list = []
        
        pid_list = list(self.patient_record.keys())
        np.random.shuffle(pid_list)
        
        fold_size = int(np.floor(len(pid_list)/nfold))
        
        for fold in range(nfold):
            print("splitting {}-fold".format(fold+1))
            fold_record = dict()
            fold_pid_list = pid_list[fold_size*fold:fold_size*(fold+1)]
            print("fold size: {}".format(len(fold_pid_list)))
        
            for pid in fold_pid_list:
                fold_record[pid] = copy.deepcopy(self.patient_record[pid])
                
            fold_record_list.append(fold_record)
        
        self.fold_record = fold_record_list

    def save_record_nfold(self, saving_dir):
        
        for n, fold_record in enumerate(self.fold_record):
            saving = saving_dir + "synthea_sinusitis_patient_record_{}fold.pkl".format(n)
            save_data(saving, fold_record)

    def save_params(self, saving_dir):
        self.params.update({"temporal_confounding_var" : self.temporal_confounding_var_idx, 
                            "fixed_confounding_var" : self.fixed_confounding_var_idx})
        saving = saving_dir + "synthea_sinusitis_params.pkl"
        save_data(saving, self.params)

    def process_record(self):
         
        code_collapsed_record = np.zeros((len(self.data_dict), len(self.code2id)))
        
        for i, (pid, code_binary) in enumerate(tqdm(self.data_dict.items())):

            new_pid = "p" + str(i) # redefine pid
            code_collapsed = torch.Tensor(np.sum(code_binary, 0))
            code_collapsed_record[i,:] = code_collapsed
            code_collapsed_binary = torch.Tensor(np.clip(code_collapsed, 0, 1))

            code_ind = []
            code_collapsed_ind = []
            for j in range(code_binary.shape[0]):
                codes = list(np.reshape(np.argwhere(code_binary[j,:] == 1), -1))
                code_collapsed_ind.extend(codes)
                codes = torch.Tensor(codes)
                codes += self.params["num_token_reserve"] # of tokens reserved for padding, SEP, CLS,...
                code_ind.append(codes.long())
            
            # code_collapsed_ind is used for PSMLP
            code_collapsed_ind = torch.Tensor(code_collapsed_ind)
            code_collapsed_ind += 1 # padding_idx=0 is reserved
            code_collapsed_ind = code_collapsed_ind.long()

            dependency_distance = calculate_dependency_distance(code_binary, self.temporal_confounding_var_idx, self.fixed_confounding_var_idx)
            propensity_score = generate_dependency_distance_treatment_prob(dependency_distance, self.params["ps_noise_var"])
            treatment_assignment = np.random.binomial(1, propensity_score.item())
            outcome = generate_dependency_distance_confounding_outcome(self.params["treatment_effect"], treatment_assignment, 
                                                     self.params["alpha"], self.params["bias"], dependency_distance, self.params["outcome_noise_var"])
            
            self.patient_record[new_pid] = {"code_binary" : code_binary, 
                                        "code_collapsed" : code_collapsed,
                                        "code_ind" : code_ind, "code_collapsed_ind" : code_collapsed_ind, 
                                        "code_collapsed_binary" : code_collapsed_binary, 
                                        "dependency_distance" : dependency_distance, "treatment_prob" : propensity_score.item(), 
                                        "treatment_assignment" : treatment_assignment, "outcome" : outcome}      

        standardizer = StandardScaler()
        code_collapsed_standardized = standardizer.fit_transform(code_collapsed_record)

        for i in range(code_collapsed_standardized.shape[0]):
            new_pid = "p" + str(i)
            self.patient_record[new_pid]["code_collapsed_LR"] = torch.Tensor(code_collapsed_standardized[i,:])      