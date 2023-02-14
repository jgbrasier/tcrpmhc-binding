import os
import sys

import numpy as np

from tqdm import tqdm

import pandas as pd
import torch
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint


from src.dataset import TCRBindDataModule, PPIDataModule, TCRpMHCDataModule
from src.models.tcr_gnn import LightningGCN

tsv = 'data/preprocessed/run329_results.tsv'
dir = '/n/data1/hms/dbmi/zitnik/lab/users/jb611/graphs/run329_results_bound'
# ckpt = 'checkpoint/run329-data/ppi_gnn/epoch=0-step=628-v1.ckpt'
run_name = 'run329-bound-data'
model_name = 'tcr_gcn'

BATCH_SIZE = 16
SEED = 24
EPOCHS = 100


data = TCRpMHCDataModule(tsv_path=tsv, processed_dir=dir, batch_size=BATCH_SIZE,\
                        id_col='uuid', y_col='binder', num_workers=16)

# npy_file =  'data/preprocessed/pan_human_data.npy'
# processed_dir =  'data/graphs/pan_human_new'
# data = PPIDataModule(npy_file=npy_file, processed_dir=processed_dir, batch_size=BATCH_SIZE)

data.setup(split='random', train_size=0.85, target='peptide', low=50, high=600, random_seed=SEED)


train_loader = data.train_dataloader()
print("Train len:", len(data.train))
test_loader = data.test_dataloader()
print("Test len:",len(data.test))


tcr_gcn = LightningGCN(embedding_dim=1280) # ESM embedding dim: 1280
checkpoint_callback = ModelCheckpoint(dirpath=os.path.join('checkpoint',run_name, model_name), save_top_k=1, monitor='val_auroc', mode='max')
tb_logger = pl_loggers.TensorBoardLogger(save_dir=os.path.join('logs',run_name), name=model_name)
trainer = pl.Trainer(max_epochs=EPOCHS, logger=tb_logger, callbacks=[checkpoint_callback], \
                    accelerator='gpu', devices=1, log_every_n_steps=30, check_val_every_n_epoch=1)
trainer.fit(tcr_gcn, train_dataloaders=train_loader, val_dataloaders=test_loader)