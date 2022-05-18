import os
import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F

import numpy as np


class TeamEmbeddingModel(nn.Module):
    def __init__(self,input_size,hidden_size):
        super(TeamEmbeddingModel, self).__init__()
        self.input_size = input_size
        self.hidden_size  = hidden_size
        self.output_size = 20
            
        self.w1 = torch.nn.Linear(2580, 1024,bias=True)
        nn.init.xavier_uniform_(self.w1.weight)
        nn.init.zeros_(self.w1.bias)
        #self.bn1 = torch.nn.BatchNorm1d(1024)
        
        self.w2 = torch.nn.Linear(1024, 512,bias=True)
        nn.init.xavier_uniform_(self.w2.weight)
        nn.init.zeros_(self.w2.bias)
        #self.bn2 = torch.nn.BatchNorm1d(512)
        
        
        self.w3 = torch.nn.Linear(512, 20,bias=True)
        nn.init.xavier_uniform_(self.w3.weight)
        nn.init.zeros_(self.w3.bias)
    

    def forward(self, x):
        x = torch.relu(self.w1(x))
        x = torch.relu(self.w2(x))
        x = self.w3(x)
        return x


