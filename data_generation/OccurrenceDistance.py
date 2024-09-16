import numpy as np
import torch
import copy
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from src.SyntheticDataGeneration import *
from src.utils import *

def calculate_occurrence_distance(binary_code, confounding_var_idx):

    # calculate the distance between the last record to last occurrence

    # this sum only works with single confounding_var
    binary_code_occurrence = binary_code[:, confounding_var_idx]
    
    try:
        last_occurrence_idx = binary_code_occurrence.nonzero()[0][-1]
        occurrence_distance = binary_code.shape[0] - last_occurrence_idx - 1
    except: # did not occur
        occurrence_distance = -1
        
    return occurrence_distance

def generate_occurrence_distance_treatment_prob(occurrence_distance, noise_var):

    noise = np.random.normal(0, noise_var)

    if occurrence_distance >= 0:

        distance_effect = np.log10(10/((occurrence_distance*5)+1))
        treatment_prob = torch.sigmoid(torch.tensor(distance_effect)) + noise
    
    else:
        treatment_prob = torch.tensor(0.1 + noise)
        
    treatment_prob = torch.clip(treatment_prob, 0.02, 0.98)
    
    return treatment_prob

def generate_occurrence_distance_confounding_outcome(treatment_effect, treatment_assignment, alpha, bias, occurrence_distance, noise_var):
    
    noise = np.random.normal(0, noise_var)
    if occurrence_distance >= 0:
        confounding_outcome = alpha * (1/(occurrence_distance+1))
    else:
        confounding_outcome = 0

    return bias + treatment_effect*treatment_assignment + confounding_outcome + noise

class EHRgeneratorOD:

    # patient record generator in occurrence distance scenario

    def __init__(self, params):
        self.params = params
        self.patient_record = dict()

    def save_record(self, saving_dir):
        saving = saving_dir + "occurrence_distance_patient_record.pkl"
        save_data(saving, self.occurrence_distance_record)
        
    def split_record_nfold(self, nfold):
        
        fold_record_list = []
        
        pid_list = list(self.occurrence_distance_record.keys())
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
            saving = saving_dir + "occurrence_distance_patient_record_{}fold.pkl".format(n)
            save_data(saving, fold_record)

    def save_params(self, saving_dir):
        self.params.update({"confounding_var" : self.confounding_var, "progression_var" : self.progression_var})
        saving = saving_dir + "occurrence_distance_params.pkl"
        save_data(saving, self.params)

    def set_variables(self):
        print("setting progression_var and confounding_var")
        selected_var_ind = np.random.choice(np.arange(self.params["record_dim"]), self.params["progression_var_num"], replace=False)
        
        self.progression_var = selected_var_ind[:self.params["progression_var_num"]]
        # confounding_var is randomly chosen from progression_var
        self.confounding_var = self.progression_var[-self.params["confounding_var_num"]:] 

        print("progression_var : {}".format(self.progression_var))
        print("confounding_var : {}".format(self.confounding_var))

    def generate_record(self, generation_num):
        
        print("generating {} records".format(generation_num))
        record_len = np.random.poisson(self.params["record_len_mean"], generation_num)
        code_collapsed_record = np.zeros((generation_num, self.params["record_dim"]))
        
        for i, rl in enumerate(tqdm(record_len)):
            
            # clip the min length of the record
            if rl < self.params["record_len_min"]:
                rl = self.params["record_len_min"]

            pid = "p" + str(i)
            intrinsic_param = np.random.uniform(self.params["alpha_range"][0], self.params["alpha_range"][1], size=self.params["record_dim"])
            intrinsic_param = np.tile(intrinsic_param, [int(rl),1]) # rl * record_dim
            time_varying_param = np.random.uniform(self.params["beta_range"][0], self.params["beta_range"][1], size=self.params["record_dim"]) # 1 * record_dim
            
            # selected variable update 
            p_list = np.random.uniform(self.params["p_range"][0], self.params["p_range"][1], self.params["progression_var_num"])
            time_varying_param[self.progression_var] = p_list
            time_varying_param = np.transpose(SplineTrendsMixture(self.params["record_dim"], rl)(np.arange(rl))) * time_varying_param # rl * record_dim
            
            code_prob = np.random.beta(intrinsic_param, time_varying_param)
            code_binary = np.random.binomial(1, code_prob)
            
            code_prob = np.reshape(code_prob, (int(rl), self.params["record_dim"]))
            code_binary = np.reshape(code_binary, (int(rl), self.params["record_dim"]))
            
            visit_check, visit_colsum = empty_visit_check(code_binary)
            
            if visit_check > 0:
                
                # check if there is a visit with 0 code and regenerate if its the case
                print("valid check...")
                zero_visit_idx = np.argwhere(visit_colsum == 0)

                for idx in zero_visit_idx:
                    random_idx = np.random.choice(self.params["record_dim"])
                    code_binary[idx,random_idx] = 1 # assign randomly chosen one code for visit that has 0 code

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

            code_collapsed_ind = torch.Tensor(code_collapsed_ind)
            code_collapsed_ind += 1 # padding_idx=0 is reserved
            code_collapsed_ind = code_collapsed_ind.long()
            
            occurrence_distance = calculate_occurrence_distance(code_binary, self.confounding_var)
            propensity_score = generate_occurrence_distance_treatment_prob(occurrence_distance, self.params["ps_noise_var"])
            treatment_assignment = np.random.binomial(1, propensity_score.item())
            outcome = generate_occurrence_distance_confounding_outcome(self.params["treatment_effect"], treatment_assignment, self.params["alpha"],
                                                  self.params["outcome_bias"], occurrence_distance, self.params["outcome_noise_var"])

            self.patient_record[pid] = {"code_prob" : code_prob, "code_binary" : code_binary, 
                                        "intrinsic_param" : intrinsic_param, "time_varying_param" : time_varying_param,
                                        "code_collapsed" : code_collapsed,
                                        "code_ind" : code_ind, "code_collapsed_ind" : code_collapsed_ind, 
                                        "code_collapsed_binary" : code_collapsed_binary, 
                                        "occurrence_distance" : occurrence_distance, "treatment_prob" : propensity_score.item(), 
                                        "treatment_assignment" : treatment_assignment, "outcome" : outcome}
            
        standardizer = StandardScaler()
        code_collapsed_standardized = standardizer.fit_transform(code_collapsed_record)

        for i in range(code_collapsed_standardized.shape[0]):
            pid = "p" + str(i)
            self.patient_record[pid]["code_collapsed_LR"] = torch.Tensor(code_collapsed_standardized[i,:])

    def update_occurrence_distance_dict(self):
        
        self.occurrence_distance_dict = dict()
        
        for pid, item in self.patient_record.items():
            
            occurrence_distance = item["occurrence_distance"]
            
            try:
                self.occurrence_distance_dict[occurrence_distance].append(pid)
            except:
                self.occurrence_distance_dict[occurrence_distance] = []
                self.occurrence_distance_dict[occurrence_distance].append(pid)

    def build_record_by_occurrence_distance(self, occurrence_distance_list, patient_num_list):
        
        assert len(occurrence_distance_list) == len(patient_num_list), "length of occurrence_sum_list and num_list must be the same"
        self.occurrence_distance_record = dict()
        
        for occurrence_distance, patient_num in zip(occurrence_distance_list, patient_num_list):
            
            print("occurrence distance {} : {} patients".format(occurrence_distance, len(self.occurrence_distance_dict[occurrence_distance])))
            
            if len(self.occurrence_distance_dict[occurrence_distance]) < patient_num:
                print("not enough patient")
                break
                
            pid_list = self.occurrence_distance_dict[occurrence_distance][:patient_num]
            np.random.shuffle(pid_list)
            
            for pid in pid_list:
                self.occurrence_distance_record[pid] = copy.deepcopy(self.patient_record[pid])