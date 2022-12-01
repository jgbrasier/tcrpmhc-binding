import sys
import os
import pandas as pd
import numpy as np

from src.dataset import TCRSeqDataset
from src.models import NetTCR

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch.optim import Optimizer, Adam

train_file = "data/sample_train.csv"
test_file = "data/sample_test.csv"

peptide_len, cdra_len, cdrb_len = 9, 30, 30

train_dataset = TCRSeqDataset(file = train_file, peptide_len=peptide_len, cdra_len=cdra_len, cdrb_len=cdrb_len)
train_size = int(len(train_dataset)*0.8)
val_size = len(train_dataset) - train_size
train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

test_dataset = TCRSeqDataset(file = test_file, test=True)

BATCH_SIZE = 16
LEARNING_RATE = 0.01
EPOCHS = 10

train_dataloader = DataLoader(train_dataset, batch_size = BATCH_SIZE)
val_dataloader = DataLoader(val_dataset, batch_size=len(val_dataset))
test_dataloader = DataLoader(test_dataset, batch_size=len(test_dataset))

DEVICE = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')

model = NetTCR(peptide_len=peptide_len, cdra_len=cdra_len, cdrb_len=cdrb_len)
criterion = nn.BCELoss()
optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

def train_epoch(idx: int, model: nn.Module, train_dataloader: DataLoader, loss_fn: torch.nn.modules.loss._Loss, optimizer: Optimizer, device: torch.device) -> float:
    last_loss = 0.0
    running_loss = 0.0
    for i, batch in enumerate(train_dataloader):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = batch
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 1000 == 999:    # print every 1000 mini-batches
            last_loss = running_loss / 1000
            running_loss = 0.0
    return last_loss

def train(model: nn.Module, train_dataloader: DataLoader, val_dataloader: DataLoader, 
        epochs: int, loss_fn: torch.nn.modules.loss._Loss, optimizer: Optimizer, device=DEVICE) -> dict:
    print("Using:", DEVICE)
    model.to(DEVICE)
    history = {'train_loss': [],
            'val_loss': []
            }
    for epoch in range(1, epochs+1):
        model.train(True)
        avg_loss = train_epoch(epoch, model, train_dataloader, loss_fn, optimizer, device)
        model.train(False)

        running_vloss = 0.0
        for i, vdata in enumerate(val_dataloader):
            vinputs, vlabels = vdata
            voutputs = model(vinputs)
            vloss = loss_fn(voutputs, vlabels)
            running_vloss += vloss

        avg_vloss = running_vloss / (i + 1)
        print("epoch: {:<20}, train loss: {:<20}, val_loss: {:<20}".format(epoch, avg_loss, avg_vloss))
        history['train_loss'].append(avg_loss)
        history['val_loss'].append(avg_vloss)

    return history

history = train(model, train_dataloader, val_dataloader, EPOCHS, criterion, optimizer)

