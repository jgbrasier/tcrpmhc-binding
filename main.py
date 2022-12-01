import sys
import os
import pandas as pd
import numpy as np

from src.dataset import TCRSeqDataset

import torch
from torch.utils.data import DataLoader
from torch.utils.data import random_split
train_file = "data_test/sample_train.csv"
test_file = "data_test/sample_test.csv"

train_dataset = TCRSeqDataset(file = train_file)
train_size = int(len(train_dataset)*0.8)
val_size = len(train_dataset) - train_size
train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

test_dataset = TCRSeqDataset(file = test_file, test=True)

BATCH_SIZE = 16

train_dataloader = DataLoader(train_dataset, batch_size = BATCH_SIZE)
val_dataloader = DataLoader(val_dataset, batch_size=len(val_dataset))
test_dataloader = DataLoader(test_dataset, batch_size=len(test_dataset))

