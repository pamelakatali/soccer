import pickle
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

import torch

#import warnings
#warnings.filterwarnings('ignore')


class GameDataset(Dataset):
    def __init__(self, data_dir):
        data = pickle.load( open( data_dir+'../train_team_data_df.pkl', 'rb' ) )
        self.cols = pickle.load( open( data_dir+'../train_team_data_cols.pkl', 'rb' ) )
        self.X = data[self.cols].values
        self.y = data['squad'].values
        
        #print(self.X.shape)
        

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        y_one_hot = np.zeros(20)#self.y[idx]
        y_one_hot[self.y[idx]] = 1
        
        return self.X[idx], y_one_hot

