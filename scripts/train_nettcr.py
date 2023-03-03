import os
import sys
import torch
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint


sys.path.insert(0, os.path.abspath(os.path.join('..')))

from src.dataset import NetTCRDataModule
from src.models import LightningNetTCR
from src.utils import blosum50_full, blosum50_20aa, enc_list_bl_max_len

PEP_LEN = 20
CDR3A_LEN = 30
CDR3B_LEN = 30

BATCH_SIZE = 32
RANDOM_SEED = 10

tcrseq = NetTCRDataModule(path_to_file="data/preprocessed/tc_hard.tsv",
                          batch_size=BATCH_SIZE)

TARGET_VALUES = ['NYNYLYRLF', 'NQKLIANQF', 'PTDNYITTY', 'MEVTPSGTWL', 'LTDEMIAQY', 'QYIKWPWYI', 'LLYDANYFL', 'RFPLTFGWCF', 'KTFPPTEPK', 'KLVALGINAV', 'RPHERNGFTVL', 'IMNDMPIYM', 'RPPIFIRRL', 'GLCTLVAML']

tcrseq.setup(sep='\t', train_size=0.85, encoder= enc_list_bl_max_len, encoding = blosum50_full, \
            peptide_len = PEP_LEN, cdra_len = CDR3A_LEN, cdrb_len = CDR3B_LEN, \
            split='hard', target="peptide", low = 40, high = 3000, random_seed=RANDOM_SEED,\
            )

train_loader = tcrseq.train_dataloader()
val_loader = tcrseq.val_dataloader()

nettcr = LightningNetTCR(peptide_len= PEP_LEN, cdra_len=CDR3A_LEN, cdrb_len=CDR3B_LEN, batch_size=tcrseq.hparams.batch_size)
checkpoint_callback = ModelCheckpoint(dirpath=os.path.join('checkpoint','run329-data', 'nettcr'), save_top_k=1, monitor='val_auroc', mode='max')
tb_logger = pl_loggers.TensorBoardLogger(save_dir=os.path.join('logs','run329-data'), name='nettcr')
trainer = pl.Trainer(max_epochs=20, logger=tb_logger, callbacks=[checkpoint_callback], log_every_n_steps=10)

trainer.fit(nettcr, train_dataloaders=train_loader, val_dataloaders=val_loader)


