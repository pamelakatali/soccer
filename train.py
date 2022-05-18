import os
import torch
from torch import nn
from torch.utils.data import Dataset

import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support

from GameModels import TeamEmbeddingModel
from GameDataset import GameDataset





def train(data_dir):

    BATCH_SIZE = 4096
    CHUNK_SIZE = 4096
    TEST_CHUNK_SIZE = 25000
    EPOCHS = 100
    EVAL_BATCH = int(CHUNK_SIZE /  BATCH_SIZE) - 10
    LR = 1e-2
    CLASS_WEIGHTS = torch.Tensor(np.ones(20))
    LOSS_FN = nn.CrossEntropyLoss(weight=CLASS_WEIGHTS)
    #MODEL_DIR = 'mlp_small'
    EVAL_BATCH = 0
    DEVICE = torch.device("cpu") 
    WEIGHT_DECAY = 0.0
    print(DEVICE)

    # Set fixed random number seed
    torch.manual_seed(42)

    # Initialize the MLP
    emb_model = TeamEmbeddingModel(2580,20)
    emb_model = emb_model.to(DEVICE)


    # Define the loss function and optimizer
    #loss_function = LOSS_FN
    optimizer = torch.optim.AdamW(emb_model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    sftmx = nn.Softmax(dim=1)
    loss_lst = []
    batch_count = 0


    # Run the training loop
    for epoch in range(0, EPOCHS): # 5 epochs at maximum
        # Print epoch
        #print(f'Starting epoch {epoch+1}')

        current_loss = 0.0

        #process chunk
        train_dataset = GameDataset(data_dir)
        # Prepare dataset
        trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

        # Iterate over the DataLoader for training data
        for i, data in enumerate(trainloader):
            emb_model.train()

            # Get inputs
            inputs, targets = data

            inputs = inputs.to(DEVICE)
            targets = targets.to(DEVICE)
            
            #print(inputs[4])
            #print(targets[4])

            # Zero the gradients
            optimizer.zero_grad()

            # Perform forward pass
            outputs = emb_model(inputs.float())

            #print(torch.reshape(targets.float(),(len(outputs),1)))
            #print(outputs)

            # Compute loss
            #print(sftmx(outputs))
            #print(outputs)
            #print(outputs.shape)
            loss = LOSS_FN(outputs, targets.float())

            # Perform backward pass
            loss.backward()

            # Perform optimization
            optimizer.step()


            # Print statistics
            current_loss += loss.item()
            batch_count += 1
            
            print('Current loss:',loss.item())
            
        #torch.save(mlp.state_dict(), data_dir+MODEL_DIR+"/mlp_mid_train_"+str(epoch)+"_epoch.pt")
        #pickle.dump(loss_lst, open( data_dir+MODEL_DIR+"/loss_lst.pkl", "wb" ))


if __name__ == '__main__':
    data_dir = '/Users/pamelakatali/Downloads/soccer_project/data/'
    train(data_dir)