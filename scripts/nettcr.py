import os
import sys
import torch
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint


sys.path.insert(0, os.path.abspath(os.path.join('..')))

from src.dataset import TCRSeqDataModule
from src.models import LightningNetTCR
from src.utils import blosum50_full, blosum50_20aa


tcrseq = TCRSeqDataModule(path_to_file='data/preprocessed/sample_train.csv',
                        peptide_len=9, cdra_len=30, cdrb_len=30, random_seed=12345)

tcrseq.setup(sep=',', encoding=blosum50_20aa, split='hard')
train_loader = tcrseq.train_dataloader()
val_loader = tcrseq.val_dataloader()

nettcr = LightningNetTCR(peptide_len=9, cdra_len=30, cdrb_len=30, batch_size=tcrseq.hparams.batch_size)
checkpoint_callback = ModelCheckpoint(dirpath=os.path.join('checkpoint','nettcr-data', 'nettcr'), save_top_k=1, monitor='val_loss')
tb_logger = pl_loggers.TensorBoardLogger(save_dir=os.path.join('logs','nettcr-data'), name='nettcr')
trainer = pl.Trainer(max_epochs=20, logger=tb_logger, callbacks=[checkpoint_callback], log_every_n_steps=10)

trainer.fit(nettcr, train_dataloaders=train_loader, val_dataloaders=val_loader)


