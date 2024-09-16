import numpy as np
import pickle
import torch
import copy
from scipy.interpolate import splev
from scipy.special import logsumexp
from tqdm import tqdm
from torch.utils.data import Dataset

class SplineTrendsMixture:
    """
    Random spline, sampled from 3 cubic splines
    """

    class BSplines:
        def __init__(self, low, high, num_bases, degree, x=None, boundaries='stack'):

            self._low = low
            self._high = high
            self._num_bases = num_bases
            self._degree = degree

            use_quantiles_as_knots = x is not None

            if use_quantiles_as_knots:
                knots = SplineTrendsMixture._quantile_knots(low, high, x, num_bases, degree)
            else:
                knots = SplineTrendsMixture._uniform_knots(low, high, num_bases, degree)

            if boundaries == 'stack':
                self._knots = SplineTrendsMixture._stack_pad(knots, degree)
            elif boundaries == 'space':
                self._knots = SplineTrendsMixture._space_pad(knots, degree)

            self._tck = (self._knots, np.eye(num_bases), degree)

        @property
        def dimension(self):
            return self._num_bases

        def design(self, x):
            print(self._tck)
            return np.array(splev(np.atleast_1d(x), self._tck)).T

    @staticmethod
    def _uniform_knots(low, high, num_bases, degree):
        num_interior_knots = num_bases - (degree + 1)
        knots = np.linspace(low, high, num_interior_knots + 2)
        return np.asarray(knots)

    @staticmethod
    def _quantile_knots(low, high, x, num_bases, degree):
        num_interior_knots = num_bases - (degree + 1)
        clipped = x[(x >= low) & (x <= high)]
        knots = np.percentile(clipped, np.linspace(0, 100, num_interior_knots + 2))
        knots = [low] + list(knots[1:-1]) + [high]
        return np.asarray(knots)

    @staticmethod
    def _stack_pad(knots, degree):
        knots = list(knots)
        knots = ([knots[0]] * degree) + knots + ([knots[-1]] * degree)
        return knots

    @staticmethod
    def _space_pad(knots, degree):
        knots = list(knots)
        d1 = knots[1] - knots[0]
        b1 = np.linspace(knots[0] - d1 * degree, knots[0], degree + 1)
        d2 = knots[-1] - knots[-2]
        b2 = np.linspace(knots[-1], knots[-1] + d2 * degree, degree + 1)
        return list(b1) + knots[1:-1] + list(b2)

    class PopulationModel:
        def __init__(self, basis, class_prob, class_coef):
            self.basis = basis
            self.n_classes = len(class_coef)

            self.class_prob = np.array(class_prob)
            self.class_coef = np.array(class_coef)

        def sample_class_prob(self, rng):
            logits = rng.normal(size=self.n_classes)
            self.class_prob[:] = np.exp(logits - logsumexp(logits))

        def sample_class_coef(self, mean, cov, rng):
            mvn_rvs = rng.multivariate_normal
            self.class_coef[:] = mvn_rvs(mean, cov, size=self.n_classes)

        def sample(self, size=1):
            z = np.random.choice(self.n_classes, size=size, p=self.class_prob)
            w = self.class_coef[z]
            return z, w

    def __init__(self, n_patients, max_time):
        class_coef = np.array([
            [1.0, 1.0, 1.0, 1.0, 1.0],  # stay
            [1.0, 1.05, 1.1, 1.15, 1.2],  # mild incline
            [1.0, 0.95, 0.9, 0.85, 0.8],  # mild decline
            [1.2, 1.15, 1.1, 1.05, 1.0], # mild decline from amplified start
            [0.8, 0.85, 0.9, 0.95, 1.0] # mild incline from amplified start
        ])
        low, high, n_bases, degree = 0.0, max_time, class_coef.shape[1], 4
        self.basis = SplineTrendsMixture.BSplines(low, high, n_bases, degree, boundaries='space')
        self.population = SplineTrendsMixture.PopulationModel(self.basis, [0.2, 0.2, 0.2, 0.2, 0.2], class_coef)
        self.classes, self.coefs = self.population.sample(size=n_patients)

    def __call__(self, time_range):
        return np.dot(self.coefs, self.basis.design(time_range).T)

class SyntheticDataset(Dataset):
    
    # build synthetic datset from data dictionary
    
    def __init__(self, data_dict):
        self.data_dict = data_dict
        self.input_keys = list(data_dict.keys())

    def __len__(self):
        return len(self.input_keys)

    def __getitem__(self,idx):
        item = self.data_dict[self.input_keys[idx]]["code_ind"]
        label = self.data_dict[self.input_keys[idx]]["treatment_assignment"]
        treatment_prob = self.data_dict[self.input_keys[idx]]["treatment_prob"]
        outcome = self.data_dict[self.input_keys[idx]]["outcome"]
        return item, label, treatment_prob, outcome, len(item), self.input_keys[idx]
    
class SyntheticDatasetCollapsed(Dataset):
    
    # build synthetic collaped datset from data dictionary for MLP
    
    def __init__(self, data_dict):
        self.data_dict = data_dict
        self.input_keys = list(data_dict.keys())

    def __len__(self):
        return len(self.input_keys)

    def __getitem__(self,idx):
        item = self.data_dict[self.input_keys[idx]]["code_collapsed_ind"]
        label = self.data_dict[self.input_keys[idx]]["treatment_assignment"]
        treatment_prob = self.data_dict[self.input_keys[idx]]["treatment_prob"]
        outcome = self.data_dict[self.input_keys[idx]]["outcome"]
        return item, label, treatment_prob, outcome, len(item), self.input_keys[idx]
    
class SyntheticDatasetCollapsedLR(Dataset):
    
    # build synthetic collaped binary datset from data dictionary for logistic regression
    
    def __init__(self, data_dict):
        self.data_dict = data_dict
        self.input_keys = list(data_dict.keys())

    def __len__(self):
        return len(self.input_keys)

    def __getitem__(self,idx):
        item = self.data_dict[self.input_keys[idx]]["code_collapsed_LR"]
        label = self.data_dict[self.input_keys[idx]]["treatment_assignment"]
        treatment_prob = self.data_dict[self.input_keys[idx]]["treatment_prob"]
        outcome = self.data_dict[self.input_keys[idx]]["outcome"]
        return item, label, treatment_prob, outcome, len(item), self.input_keys[idx]
    
def calculate_code_percentile(data_record, percentiles, code_dim):
    
    code_prevalence = np.zeros((len(data_record), code_dim))
    
    for i, (k, v) in enumerate(data_record.items()):
        code_prevalence[i] = np.sum(v["code_binary"], 0) / v["code_binary"].shape[0]
    
    return np.percentile(code_prevalence, percentiles, axis=0)

def convert_HDPS_data_record(data_record, code_dim):

    HDPS_data_record = dict()
    p50, p75 = calculate_code_percentile(data_record, [50, 75], code_dim)
    
    for pid, v in tqdm(data_record.items()):
        
        HDPS_data_record[pid] = copy.deepcopy(v)

        code_prevalence = np.sum(v["code_binary"], 0) / v["code_binary"].shape[0]
        at_least_one = np.sum(v["code_binary"], 0) >= 1.
        above_p50 = code_prevalence > p50
        above_p75 = code_prevalence > p75
        hdps_var = np.concatenate([at_least_one, above_p50, above_p75]) * 1 # binary vector
        
        HDPS_data_record[pid]["code_collapsed_ind"] = torch.argwhere(torch.tensor(hdps_var)).flatten() + 1 # padding_idx = 0
        HDPS_data_record[pid]["code_collapsed_LR"] = torch.Tensor(hdps_var)

    return HDPS_data_record

def check_zero_record(data_record):
    
    for k, v in data_record.items():
        mysum = np.sum(v["code_binary"])
        if mysum ==0:
            print("{} zero records detected".format(mysum))

def check_record_len_stat(data_record):

    record_len_list = []

    for k, v in data_record.items():
        record_len_list.append(len(v["code_ind"]))

    print("max record length: {}".format(np.max(record_len_list)))
    print("min record length: {}".format(np.min(record_len_list)))
    print("avg record length: {}".format(np.mean(record_len_list)))

def check_treatment_stat(data_record):
    
    t_prob_sum = 0.
    t_sum = 0.
    one_prob_sum = 0.
    zero_prob_sum = 0.
    
    for k, v in data_record.items():
        t_prob_sum += v["treatment_prob"]
        t_sum += v["treatment_assignment"]

        if v["treatment_assignment"] == 1:
            one_prob_sum += v["treatment_prob"]
        else:
            zero_prob_sum += v["treatment_prob"]

        
    print("avg treatment probability: {}".format(t_prob_sum / len(data_record)))
    print("sum of treatment assignment: {}".format(t_sum))
    print("avg treatment probability calculated by treatment assignment: {}".format(t_sum / len(data_record)))
    print("avg treatment probability of treated individuals: {}".format(one_prob_sum/t_sum))
    print("avg treatment probability of control individuals: {}".format(zero_prob_sum/(len(data_record)-t_sum)))

def check_data_quality(data_record):
    
    # check if the data have zero length record
    check_zero_record(data_record)
    
    # check the average treatment probability
    check_treatment_stat(data_record)

    # check the max, min, avg of record length
    check_record_len_stat(data_record)

def random_split_data_record(data_record, training_validation_test_num):
    
    training_data_record = dict()
    validation_data_record = dict()
    test_data_record = dict()
    
    keys = list(data_record.keys())
    training_size = training_validation_test_num[0]
    validation_size = training_validation_test_num[1]
    test_size = training_validation_test_num[2]
    np.random.shuffle(keys)
    
    training_data_keys = keys[:training_size]
    validation_data_keys = keys[training_size:(training_size+validation_size)]
    test_data_keys = keys[(training_size+validation_size):]
    
    for k in training_data_keys:
        training_data_record[k] = data_record[k]
    
    for k in validation_data_keys:
        validation_data_record[k] = data_record[k]
    
    for k in test_data_keys:
        test_data_record[k] = data_record[k]
    
    print("training data size: {}".format(len(training_data_record)))
    check_data_quality(training_data_record)
    print("validation data size: {}".format(len(validation_data_record)))
    check_data_quality(validation_data_record)
    print("test data size: {}".format(len(test_data_record)))
    check_data_quality(test_data_record)
    
    return training_data_record, validation_data_record, test_data_record