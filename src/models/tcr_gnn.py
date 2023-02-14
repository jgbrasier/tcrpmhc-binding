# Building model
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
from torch_geometric.utils import dropout_adj
from torch.optim.lr_scheduler import MultiStepLR

import pytorch_lightning as pl

from torchmetrics.classification.accuracy import BinaryAccuracy
from torchmetrics.classification.auroc import BinaryAUROC
from torchmetrics.classification.precision_recall import BinaryPrecision, BinaryRecall
from torchmetrics.classification.f_beta import BinaryF1Score

class GCN(nn.Module):
    def __init__(self, n_output=1, embedding_dim= 1280, output_dim=128, dropout=0.2):
        super(GCN, self).__init__()
        print('GCNN Loaded')

        # for protein 1
        self.n_output = n_output
        self.conv1 = GCNConv(embedding_dim, embedding_dim//2)
        self.conv2 = GCNConv(embedding_dim//2, embedding_dim)
        self.fc1 = nn.Linear(embedding_dim, output_dim)


        self.relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(dropout)

        # combined layers
        self.fc2 = nn.Linear(output_dim ,output_dim//2)
        self.out = nn.Linear(output_dim//2, self.n_output)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = self.relu(x)
        x = self.conv2(x, edge_index)
        x = self.relu(x)

	    # global pooling
        x = global_mean_pool(x, batch)   

        # flatten
        xf = self.relu(self.fc1(x))
        # x = self.dropout(x)

        # add some dense layers
        xf = self.fc2(xf)
        xf = self.relu(xf)
        # xf = self.dropout(xf)
        out = self.out(xf)
        return out

class LightningGCN(pl.LightningModule):
    def __init__(self, n_output=1, embedding_dim = 1280, output_dim=128, dropout=0.2, \
                learning_rate = 0.001,):
        super().__init__()
        self.save_hyperparameters()

        # for protein 1
        self.model = GCN(n_output=1, embedding_dim=embedding_dim, output_dim=output_dim, dropout=dropout)
        self.sigmoid = nn.Sigmoid()

        # metrics
        self.loss_fn = nn.BCELoss()
        self._acc = BinaryAccuracy()
        self._precision = BinaryPrecision()
        self._recall = BinaryRecall()
        self._f1 = BinaryF1Score()
        self._auroc = BinaryAUROC()

    def forward(self, data):
        #get graph input for protein 1 
        x, edge_index, batch = data.x, data.edge_index, data.batch
        # get graph input for protein 2
        x = self.model(x, edge_index, batch)
        out = self.sigmoid(x)
        return out

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

    def training_step(self, batch, batch_idx):
        prot, label = batch
        label = label.type(torch.float) # output is float32 needs to match
        output = self(prot)
        loss = self.loss_fn(output, label)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=False, logger=True, batch_size=len(batch))
        return loss

    def validation_step(self, batch, batch_idx):
        prot, label = batch
        output = self(prot)
        label = label.type(torch.float)
        loss = self.loss_fn(output, label) # output is float32 needs to match
        acc = self._acc(output, label)
        precision = self._precision(output, label)
        recall = self._recall(output, label)
        f1 = self._f1(output, label)
        auroc = self._auroc(output, label)
        self.log("val_acc", acc, on_step=True, on_epoch=True, prog_bar=False, logger=True, batch_size=len(batch))
        self.log("val_precision", precision, on_step=True, on_epoch=True, prog_bar=False, logger=True, batch_size=len(batch))
        self.log("val_recall", recall, on_step=True, on_epoch=True, prog_bar=False, logger=True, batch_size=len(batch))
        self.log("val_f1", f1, on_step=True, on_epoch=True, prog_bar=False, logger=True, batch_size=len(batch))
        self.log("val_auroc", auroc, on_step=True, on_epoch=True, prog_bar=False, logger=True, batch_size=len(batch))
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=False, logger=True, batch_size=len(batch))
        return loss
    
    def test_step(self, batch, batch_idx):
        prot, label = batch
        output = self(prot)
        loss = self.loss_fn(output, label)
        acc = self._acc(output, label)
        precision = self._precision(output, label)
        recall = self._recall(output, label)
        f1 = self._f1(output, label)
        auroc = self._auroc(output, label)
        self.log("val_acc", acc, on_step=True, on_epoch=True, prog_bar=False, logger=True, batch_size=len(batch))
        self.log("val_precision", precision, on_step=True, on_epoch=True, prog_bar=False, logger=True, batch_size=len(batch))
        self.log("val_recall", recall, on_step=True, on_epoch=True, prog_bar=False, logger=True, batch_size=len(batch))
        self.log("val_f1", f1, on_step=True, on_epoch=True, prog_bar=False, logger=True, batch_size=len(batch))
        self.log("val_auroc", auroc, on_step=True, on_epoch=True, prog_bar=False, logger=True, batch_size=len(batch))
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=False, logger=True, batch_size=len(batch))
        return loss    