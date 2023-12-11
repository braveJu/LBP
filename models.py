import torch.nn as nn
from utils import HP
import numpy as np

class DongjuModel(nn.Module):
    def __init__(self, output_size=HP['output_size']):
        super(DongjuModel, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(256, 100),
            nn.ReLU(),
            nn.Linear(100, output_size),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        output = self.linear(x)
        return output
