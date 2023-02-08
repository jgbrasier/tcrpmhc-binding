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

from src.dataset import TCRBindDataModule, TCRpMHCDataModule
from src.models import LightningGCNN, LightningGCN

tsv = 'data/preprocessed/run329_results.tsv'
dir = 'data/graphs/run329_results_bound'
ckpt = 'checkpoint/run329-data/ppi_gnn/epoch=0-step=628-v1.ckpt'
run_name = 'run329-bound-data'

BATCH_SIZE = 16
SEED = 42
EPOCHS = 100


# data = TCRBindDataModule(tsv_path=tsv, processed_dir=dir, batch_size=BATCH_SIZE,\
#                         target='peptide', low=50, high=800)
                    
# data.setup(train_size=0.85, random_seed=SEED)


data = TCRpMHCDataModule(tsv_path=tsv, processed_dir=dir, batch_size=BATCH_SIZE,\
                        id_col='uuid', y_col='binder')

data.setup(train_size=0.85, target='peptide', low=50, high=700, random_seed=SEED)

train_loader = data.train_dataloader()
print("Train len:", len(data.train))
test_loader = data.test_dataloader()
print("Test len:",len(data.test))

# for batch in train_loader:
#     prot1, prot2, label = batch
#     print(prot2.x)
#     break

# ppigcnn = LightningGCNN(embedding_dim=1280, learning_rate=0.001) # .load_from_checkpoint(ckpt) # ESM embedding dim: 1280
gcn = LightningGCN(embedding_dim=1280, learning_rate=0.001)
checkpoint_callback = ModelCheckpoint(dirpath=os.path.join('checkpoint',run_name, 'gcn'), save_top_k=1, monitor='val_auroc', mode='max')
tb_logger = pl_loggers.TensorBoardLogger(save_dir=os.path.join('logs',run_name), name='gcn')
trainer = pl.Trainer(max_epochs=EPOCHS, logger=tb_logger, callbacks=[checkpoint_callback], log_every_n_steps=10, check_val_every_n_epoch=1)
trainer.fit(gcn, train_dataloaders=train_loader, val_dataloaders=test_loader)

# ppigcnn.eval()
# for batch in train_loader:
#     prot1, prot2, label = batch
#     with torch.no_grad():
#         y_hat = ppigcnn(prot1, prot2)
#         print(y_hat, label)