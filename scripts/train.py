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



from src.dataset import TCRBindDataModule, PPIDataModule
from src.models.ppi_gnn import LightningGCNN, GCNN, AttGNN

tsv = 'data/preprocessed/run329_results.tsv'
dir = 'data/graphs/run329_results'
ckpt = 'checkpoint/run329-data/ppi_gnn/epoch=0-step=628-v1.ckpt'
run_name = 'run329-data'

BATCH_SIZE = 4
SEED = 42
EPOCHS = 100


data = TCRBindDataModule(tsv_path=tsv, processed_dir=dir, batch_size=BATCH_SIZE,\
                        y_col='binder', target='peptide', low=50, high=700)

# npy_file =  'data/preprocessed/pan_human_data.npy'
# processed_dir =  'data/graphs/pan_human_new'
# data = PPIDataModule(npy_file=npy_file, processed_dir=processed_dir, batch_size=BATCH_SIZE)

data.setup(train_size=0.85, random_seed=SEED)


train_loader = data.train_dataloader()
print("Train len:", len(data.train))
test_loader = data.test_dataloader()
print("Test len:",len(data.test))


model = AttGNN(embedding_dim=1280)
criterion = nn.BCELoss()
optimizer = Adam(model.parameters(), lr=0.001)

def train_epoch(idx: int, model: nn.Module, train_dataloader: DataLoader, loss_fn: torch.nn.modules.loss._Loss, optimizer: Optimizer, device: torch.device) -> float:
    last_loss = 0.0
    running_loss = 0.0

    print_every = 10

    for i, batch in enumerate(train_dataloader):
        # get the inputs; data is a list of [inputs, labels]
        prot1, prot2, labels = batch
        # zero the parameter gradients
        optimizer.zero_grad()

        # a = list(model.parameters())[0].clone()

        # forward + backward + optimize
        outputs = model(prot1, prot2)
        labels = labels.type(torch.float)
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
            vprot1, vprot2, vlabels = vdata
            voutputs = model(vprot1, vprot2)
            # print(voutputs, vlabels)
            vlabels = vlabels.type(torch.float)
            vloss = loss_fn(voutputs, vlabels)
            running_vloss += vloss

        avg_vloss = running_vloss / (i + 1)
        print("epoch: {:<5} train loss: {:<20} val_loss: {:<20}".format(epoch, avg_loss, avg_vloss))
        history['train_loss'].append(avg_loss)
        history['val_loss'].append(avg_vloss)

    return history

history = train(model, train_loader, test_loader, EPOCHS, criterion, optimizer)

