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

from src.dataset import TCRBindDataModule
from src.models import LightningGCNN

tsv = 'data/preprocessed/iedb_3d_binding.tsv'
dir = 'data/graphs/iedb_3d_no_b2m'
ckpt = 'checkpoint/pan-human-data/ppi_gnn/epoch=18-step=41705.ckpt'

BATCH_SIZE = 4
SEED = 42
EPOCHS = 100


data = TCRBindDataModule(tsv_path=tsv, processed_dir=dir, batch_size=BATCH_SIZE,\
                        low=2, high=10)
                    
data.setup(train_size=0.8, random_seed=SEED)

train_loader = data.train_dataloader()
print("Train len:", len(data.train))
test_loader = data.test_dataloader()
print("Test len:",len(data.test))

ppigcnn = LightningGCNN().load_from_checkpoint(ckpt) # ESM embedding dim: 1280
checkpoint_callback = ModelCheckpoint(dirpath=os.path.join('checkpoint','iedb-3d-data', 'ppi_gnn'), save_top_k=1, monitor='val_auroc', mode='max')
tb_logger = pl_loggers.TensorBoardLogger(save_dir=os.path.join('logs','iedb-3d-data'), name='ppi_gnn')
trainer = pl.Trainer(max_epochs=EPOCHS, logger=tb_logger, callbacks=[checkpoint_callback], log_every_n_steps=10, check_val_every_n_epoch=1)
trainer.fit(ppigcnn, train_dataloaders=train_loader, val_dataloaders=test_loader)