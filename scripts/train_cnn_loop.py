import sys
import os
import pandas as pd
import numpy as np
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch.optim import Optimizer, Adam

# DEVICE = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
DEVICE = torch.device('cpu')
print("Using:", DEVICE)


from src.dataset import ImageClassificationDataModule
from src.models.tcr_cnn import ResNet, SimpleCNN, ResNet50TransferLearning

tsv = 'data/preprocessed/run329_results.tsv'
dir = '/n/data1/hms/dbmi/zitnik/lab/users/jb611/dist_mat/run329_results_bound'
run_name = 'run329-bound-data'

BATCH_SIZE = 1
SEED = 42
EPOCHS = 100


data = ImageClassificationDataModule(tsv_path=tsv, processed_dir=dir, batch_size=BATCH_SIZE,\
                        id_col='uuid', y_col='binder')

# npy_file =  'data/preprocessed/pan_human_data.npy'
# processed_dir =  'data/graphs/pan_human_new'
# data = PPIDataModule(npy_file=npy_file, processed_dir=processed_dir, batch_size=BATCH_SIZE)

data.setup(train_size=0.85, target='peptide', low=50, high=700, random_seed=SEED)

df = pd.read_csv(tsv, sep='\t')

train_loader = data.train_dataloader()
print("Train len:", len(data.train))
test_loader = data.test_dataloader()
print("Test len:",len(data.test))

# for batch in train_loader:
#     name, _, label = batch
#     print(name, label)

# model = ResNet([2, 2], num_classes=1).float()
model = SimpleCNN((1, 427, 427), 1)
criterion = nn.BCELoss()
optimizer = Adam(model.parameters(), lr=0.01)

def train_epoch(idx: int, model: nn.Module, train_dataloader: DataLoader, loss_fn: torch.nn.modules.loss._Loss, optimizer: Optimizer, device: torch.device) -> float:
    last_loss = 0.0
    running_loss = 0.0

    print_every = 10

    for i, batch in enumerate(train_dataloader):
        # get the inputs; data is a list of [inputs, labels]
        # prot1, prot2, labels = batch
        img, labels = batch
        # zero the parameter gradients
        optimizer.zero_grad()

        # a = list(model.parameters())[0].clone()

        # forward + backward + optimize
        outputs = model(img)
        labels = labels.float().unsqueeze(1)
        # print(outputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        # check if weights are being updated
        # b = list(model.parameters())[0].clone()
        # print(torch.equal(a, b))
        # assert torch.equal(a, b)

        # print statistics
        running_loss += loss.item()
        if i % print_every == print_every-1:    # print every 100 mini-batches
            last_loss = running_loss / print_every
            running_loss = 0.0
    return last_loss

def train(model: nn.Module, train_dataloader: DataLoader, val_dataloader: DataLoader, 
        epochs: int, loss_fn: torch.nn.modules.loss._Loss, optimizer: Optimizer, device=DEVICE) -> dict:
    history = {'train_loss': [],
            'val_loss': []
            }
    for epoch in range(1, epochs+1):
        model.train(True)
        t1 = time.time()
        avg_loss = train_epoch(epoch, model, train_dataloader, loss_fn, optimizer, device)
        t2 = time.time()

        print(f'epoch {epoch} time: {t2-t1}')
        model.train(False)

        running_vloss = 0.0
        for i, vdata in enumerate(val_dataloader):
            # vprot1, vprot2, vlabels = vdata
            # voutputs = model(vprot1, vprot2)
            vimg, vlabels = vdata
            voutputs = model(vimg)

            print(voutputs, vlabels)
            vlabels = vlabels.float().unsqueeze(1)
            vloss = loss_fn(voutputs, vlabels)
            running_vloss += vloss

        avg_vloss = running_vloss / (i + 1)
        print("epoch: {:<5} train loss: {:<20} val_loss: {:<20}".format(epoch, avg_loss, avg_vloss))
        history['train_loss'].append(avg_loss)
        history['val_loss'].append(avg_vloss)

    return history

history = train(model, train_loader, test_loader, EPOCHS, criterion, optimizer)

