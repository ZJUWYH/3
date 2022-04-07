import os
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
# from tqdm.notebook import tqdm
from collections import Counter
import torch.nn as nn
import gc
import time
import torch
import numpy as np
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
import random
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pack_sequence, pad_packed_sequence
# from DNN_model import GRUAutoEncoder, CustomModel, LSTMAutoEncoder
import json
import copy
from tqdm import tqdm
import scipy


def save_tuple(data, path):
    with open(path, 'w') as f_json:
        json.dump(data, f_json)


def read_tuple(path):
    with open(path, 'r') as f_json:
        data = json.load(f_json)
    return data


def set_seed(seed):
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except Exception as e:
        print("Set seed failed,details are ", e)
        pass
    import numpy as np
    np.random.seed(seed)
    import random as python_random
    python_random.seed(seed)


def generate_rand_psd(matrixSize):
    A = np.random.rand(matrixSize, matrixSize)
    B = np.dot(A, A.transpose())
    return B


class Preprocessing:
    def drop_sensors(df, sensor_index):
        df0 = df.copy()
        df0.drop(df0.columns[sensor_index], axis=1, inplace=True)
        return df0

    def drop_units(df, unit_index):
        df0 = df.copy()
        df0.drop(df0[df0[df0.columns[0]].isin(unit_index)].index, axis=0, inplace=True)
        return df0.reset_index(drop=True)

    def add_timeseries(df):
        df0 = df.copy()
        df0["Time"] = df0.groupby(["Unit"]).cumcount() + 1
        return df0

def rmse(predictions, targets):
    return np.sqrt(np.mean((predictions-targets)**2))


class AircraftDataset(Dataset):
    def __init__(self, df, labels):  # df is a dataframe and label is an array indicate the true failure mode
        self.df = df.groupby("Unit").agg(list).reset_index()
        self.labels = labels

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        data = {}
        #         sensor = ['T24', 'T30', 'T50', 'P30', 'Nf', 'Nc', 'Ps30',
        #                   'phi', 'NRf', 'NRc', 'BPR', 'htBleed', 'W31', 'W32']
        sensor = ['T24', 'T30', 'T50', 'P30', 'Ps30', 'phi']
        dai_hao = ["s1","s2","s3","s4","s5","s6"]
        multi_sensor = []
        for dai,sensor_name in zip(dai_hao,sensor):
            multi_sensor.append(np.array(self.df[sensor_name].values.tolist()[idx]))
            single_sensor = np.array(self.df[sensor_name].values.tolist()[idx],dtype=np.float64)[:, None]
            data[dai] = single_sensor
        multi_sensor = np.vstack(multi_sensor).transpose(1, 0)
        data["input"] = np.array(multi_sensor, dtype=np.float64)
        data["lifetime"] = np.array(len(multi_sensor), dtype=np.int64)
        g = self.df["Time"].values.tolist()[idx]
        data["Phi_l"] = np.array([np.array([1, i / 500, (i / 500) * (i / 500)]) for i in g], dtype=np.float64)
        # data["Phi_l"] = np.array([np.array([1, i, i * i], dtype=np.int64) for i in g], dtype=np.int64)
        if self.labels[idx].item() == -1:
            data["mode"] = np.array([1, 0], dtype=np.float64)
        elif self.labels[idx].item() == 1:
            data["mode"] = np.array([0, 1], dtype=np.float64)
        return data


class AircraftDataset_one_mode(AircraftDataset):
    def __init__(self, df, labels, mode):  # df is a dataframe and label is an array indicate the true failure mode
        super().__init__(df, labels)
        self.mode = mode
        self.mode_index = np.where(self.labels.reshape(-1) == self.mode)[0].tolist()

    def __len__(self):
        return len(self.mode_index)

    def __getitem__(self, idx):
        mode_index = self.mode_index[idx]
        return super().__getitem__(mode_index)
