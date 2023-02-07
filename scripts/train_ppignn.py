import os
import sys

import numpy as np

from tqdm.notebook import tqdm

import pandas as pd
import torch
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint

# sys.path.insert(0, os.path.abspath(os.path.join('..')))

from src.dataset import PPIDataModule
from src.models import LightningGCNN

npy_file =  'data/preprocessed/pan_human_data.npy'
processed_dir =  'data/graphs/pan_human_new'
BATCH_SIZE = 8
SEED = 20
EPOCHS = 50
ppi_data = PPIDataModule(npy_file=npy_file, processed_dir=processed_dir, batch_size=BATCH_SIZE)
ppi_data.setup(train_size=0.85, random_seed=SEED)

train_loader = ppi_data.train_dataloader()
print("Train len:", len(ppi_data.train))
test_loader = ppi_data.test_dataloader()
print("Test len:",len(ppi_data.test))


ppigcnn = LightningGCNN(num_features_pro=1280) # ESM embedding dim: 1280
checkpoint_callback = ModelCheckpoint(dirpath=os.path.join('checkpoint','pan-human-data', 'ppi_gnn'), save_top_k=1, monitor='val_auroc', mode='max')
tb_logger = pl_loggers.TensorBoardLogger(save_dir=os.path.join('logs','pan-human-data'), name='ppi_gnn')
trainer = pl.Trainer(max_epochs=EPOCHS, logger=tb_logger, callbacks=[checkpoint_callback], log_every_n_steps=10, check_val_every_n_epoch=1)
trainer.fit(ppigcnn, train_dataloaders=train_loader, val_dataloaders=test_loader)