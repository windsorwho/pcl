import torch
from torch.utils.data import Dataset
import numpy as np
import logging
import os
import utils


class PCLTrainDataset(Dataset):
    def __init__(self, npz_path):
        if not os.path.exists(npz_path):
            logging.error('NPZ file does not exist. Will throw exception.')
            raise ValueError
        np_dict = np.load(npz_path)
        self.features = torch.from_numpy(
            utils.normalize_features(np_dict['features']))
        self.labels = torch.from_numpy(np_dict['labels'])
        assert len(self.labels) == self.features.shape[0]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]
