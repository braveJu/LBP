import torch
import torch.utils.data as data
from utils import preprocess, onehot_encoding
import numpy as np


class KTHDataset(data.Dataset):
    def __init__(self, x_data, y_data):
        super(KTHDataset, self).__init__()
        self.x = x_data
        self.y = y_data

    def __getitem__(self, index):
        x = preprocess(self.x[index])
        x = torch.Tensor(x.astype(np.float16))
        y = onehot_encoding(self.y[index])
        return x, y

    def __len__(self):
        return len(self.x)
