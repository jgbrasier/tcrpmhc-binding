import numpy as np
import torch

import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, Conv1d

import pytorch_lightning as pl

from torchmetrics.classification.accuracy import BinaryAccuracy
from torchmetrics.classification.auroc import BinaryAUROC
from torchmetrics.classification.precision_recall import BinaryPrecision, BinaryRecall
from torchmetrics.classification.f_beta import BinaryF1Score


# import pytorch_lightning as pl

class GlobalMaxPool1D(nn.Module):
    def __init__(self, data_format='features_last'):
        super(GlobalMaxPool1D, self).__init__()
        self.data_format = data_format
        self.step_axis = 2 if self.data_format == 'features_last' else 1

    def forward(self, input):
        return torch.max(input, axis=self.step_axis).values

class LightningNetTCR(pl.LightningModule):
    def __init__(self, peptide_len: int, cdra_len: int,cdrb_len: int, batch_size=16, 
                n_kernels: int = 5, n_filters: int = 16, hidden_dim: int = 32, device='cpu'):
        super().__init__()

        self.save_hyperparameters()
        
        self.pep_conv = nn.ModuleList([Conv1d(in_channels=self.hparams.peptide_len, out_channels=self.hparams.n_filters, kernel_size=2*i+1, padding='same', device=self.hparams.device) for i in range(self.hparams.n_kernels)])
        self.cdra_conv = nn.ModuleList([Conv1d(in_channels=self.hparams.cdra_len, out_channels=self.hparams.n_filters, kernel_size=2*i+1, padding='same', device=self.hparams.device) for i in range(self.hparams.n_kernels)])
        self.cdrb_conv = nn.ModuleList([Conv1d(in_channels=self.hparams.cdrb_len, out_channels=self.hparams.n_filters, kernel_size=2*i+1, padding='same', device=self.hparams.device) for i in range(self.hparams.n_kernels)])

        self.pool =GlobalMaxPool1D()
        self.activation = nn.Sigmoid()

        self.lin1 = Linear(in_features=3*self.hparams.n_kernels*n_filters, out_features=self.hparams.hidden_dim, device=self.hparams.device)
        self.lin2 = Linear(in_features=self.hparams.hidden_dim, out_features=1, device=self.hparams.device)

        self.criterion = nn.BCELoss()

        self.acc = BinaryAccuracy()
        self.precision = BinaryPrecision()
        self.recall = BinaryRecall()
        self.f1 = BinaryF1Score()
        self.auroc = BinaryAUROC()

    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.01)

    def forward(self, x):
        x_peptide, x_cdra, x_cdrb = x

        pep_out_list = [self.pool(self.activation(self.pep_conv[n](x_peptide))) for n in range(self.hparams.n_kernels)]
        cdra_out_list = [self.pool(self.activation(self.cdra_conv[n](x_cdra))) for n in range(self.hparams.n_kernels)]
        cdrb_out_list = [self.pool(self.activation(self.cdrb_conv[n](x_cdrb))) for n in range(self.hparams.n_kernels)]

        pep_out = torch.cat(pep_out_list, dim=1)
        cdra_out = torch.cat(cdra_out_list, dim=1)
        cdrb_out = torch.cat(cdrb_out_list, dim=1)

        out = torch.cat([pep_out, cdra_out, cdrb_out], dim=1)

        out = self.activation(self.lin1(out))
        out = self.activation(self.lin2(out))
        return out

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        y = torch.unsqueeze(y, dim=1)
        loss = self.criterion(y_hat, y)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=False, logger=True, batch_size=len(batch))
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        y = torch.unsqueeze(y, dim=1)
        loss = self.criterion(y_hat, y)
        acc = self.acc(y_hat, y)
        precision = self.precision(y_hat, y)
        recall = self.recall(y_hat, y)
        f1 = self.f1(y_hat, y)
        auroc = self.auroc(y_hat, y)
        self.log("accuracy", acc, on_step=True, on_epoch=True, prog_bar=False, logger=True, batch_size=len(batch))
        self.log("precision", precision, on_step=True, on_epoch=True, prog_bar=False, logger=True, batch_size=len(batch))
        self.log("recall", recall, on_step=True, on_epoch=True, prog_bar=False, logger=True, batch_size=len(batch))
        self.log("f1", f1, on_step=True, on_epoch=True, prog_bar=False, logger=True, batch_size=len(batch))
        self.log("auroc", auroc, on_step=True, on_epoch=True, prog_bar=False, logger=True, batch_size=len(batch))
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=False, logger=True, batch_size=len(batch))
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        y = torch.unsqueeze(y, dim=1)
        loss = self.criterion(y_hat, y)
        acc = self.acc(y_hat, y)
        precision = self.precision(y_hat, y)
        recall = self.recall(y_hat, y)
        f1 = self.f1(y_hat, y)
        auroc = self.auroc(y_hat, y)
        self.log("accuracy", acc, on_step=True, on_epoch=True, prog_bar=False, logger=True, batch_size=len(batch))
        self.log("precision", precision, on_step=True, on_epoch=True, prog_bar=False, logger=True, batch_size=len(batch))
        self.log("recall", recall, on_step=True, on_epoch=True, prog_bar=False, logger=True, batch_size=len(batch))
        self.log("f1", f1, on_step=True, on_epoch=True, prog_bar=False, logger=True, batch_size=len(batch))
        self.log("auroc", auroc, on_step=True, on_epoch=True, prog_bar=False, logger=True, batch_size=len(batch))
        self.log("test_loss", loss, on_step=True, on_epoch=True, prog_bar=False, logger=True, batch_size=len(batch))
        return loss

class NetTCR(nn.Module):
    def __init__(self, peptide_len: int, cdra_len: int,cdrb_len: int, batch_size=16, device='cpu'):
        super().__init__()

        self._elems = {'peptide': peptide_len, 'cdra': cdra_len, 'cdrb': cdrb_len}
        self.batch_size = batch_size

        self.n_kernels = 5
        n_filters = 16
        hidden_dim = 32

        self.pep_conv = nn.ModuleList([Conv1d(in_channels=peptide_len, out_channels=n_filters, kernel_size=2*i+1, padding='same', device=device) for i in range(self.n_kernels)])
        self.cdra_conv = nn.ModuleList([Conv1d(in_channels=cdra_len, out_channels=n_filters, kernel_size=2*i+1, padding='same', device=device) for i in range(self.n_kernels)])
        self.cdrb_conv = nn.ModuleList([Conv1d(in_channels=cdrb_len, out_channels=n_filters, kernel_size=2*i+1, padding='same', device=device) for i in range(self.n_kernels)])

        self.pool =GlobalMaxPool1D()
        self.activation = nn.Sigmoid()

        self.lin1 = Linear(in_features=3*self.n_kernels*n_filters, out_features=hidden_dim, device=device)
        self.lin2 = Linear(in_features=hidden_dim, out_features=1, device=device)
        
    def forward(self, x):
        x_peptide, x_cdra, x_cdrb = x

        pep_out_list = [self.pool(self.activation(self.pep_conv[n](x_peptide))) for n in range(self.n_kernels)]
        cdra_out_list = [self.pool(self.activation(self.cdra_conv[n](x_cdra))) for n in range(self.n_kernels)]
        cdrb_out_list = [self.pool(self.activation(self.cdrb_conv[n](x_cdrb))) for n in range(self.n_kernels)]

        pep_out = torch.cat(pep_out_list, dim=1)
        cdra_out = torch.cat(cdra_out_list, dim=1)
        cdrb_out = torch.cat(cdrb_out_list, dim=1)

        # pep_out_1 = self.activation(self.pep_conv[0](x_peptide))
        # pep_out_1 = self.pool(pep_out_1)
        # pep_out_3 = self.activation(self.pep_conv[1](x_peptide))
        # pep_out_3 = self.pool(pep_out_3)
        # pep_out_5 = self.activation(self.pep_conv[2](x_peptide))
        # pep_out_5 = self.pool(pep_out_5)
        # pep_out_7 = self.activation(self.pep_conv[3](x_peptide))
        # pep_out_7 = self.pool(pep_out_7)
        # pep_out_9 = self.activation(self.pep_conv[4](x_peptide))
        # pep_out_9 = self.pool(pep_out_9)

        # cdra_out_1 = self.activation(self.cdra_conv[0](x_cdra))
        # cdra_out_1 = self.pool(cdra_out_1)
        # cdra_out_3 = self.activation(self.cdra_conv[1](x_cdra))
        # cdra_out_3 = self.pool(cdra_out_3)
        # cdra_out_5 = self.activation(self.cdra_conv[2](x_cdra))
        # cdra_out_5 = self.pool(cdra_out_5)
        # cdra_out_7 = self.activation(self.cdra_conv[3](x_cdra))
        # cdra_out_7 = self.pool(cdra_out_7)
        # cdra_out_9 = self.activation(self.cdra_conv[4](x_cdra))
        # cdra_out_9 = self.pool(cdra_out_9)

        # cdrb_out_1 = self.activation(self.cdrb_conv[0](x_cdrb))
        # cdrb_out_1 = self.pool(cdrb_out_1)
        # cdrb_out_3 = self.activation(self.cdrb_conv[1](x_cdrb))
        # cdrb_out_3 = self.pool(cdrb_out_3)
        # cdrb_out_5 = self.activation(self.cdrb_conv[2](x_cdrb))
        # cdrb_out_5 = self.pool(cdrb_out_5)
        # cdrb_out_7 = self.activation(self.cdrb_conv[3](x_cdrb))
        # cdrb_out_7 = self.pool(cdrb_out_7)
        # cdrb_out_9 = self.activation(self.cdrb_conv[4](x_cdrb))
        # cdrb_out_9 = self.pool(cdrb_out_9)

        # pep_out = torch.cat([pep_out_1, pep_out_3, pep_out_5, pep_out_7, pep_out_9], dim=1)
        # cdra_out = torch.cat([cdra_out_1, cdra_out_3, cdra_out_5, cdra_out_7, cdra_out_9], dim=1)
        # cdrb_out = torch.cat([cdrb_out_1, cdrb_out_3, cdrb_out_5, cdrb_out_7, cdrb_out_9], dim=1)

        out = torch.cat([pep_out, cdra_out, cdrb_out], dim=1)

        out = self.activation(self.lin1(out))
        out = self.activation(self.lin2(out))

        return out


if __name__=='__main__':
    device = torch.device('mps') if torch.backends.mps.is_available() else 'cpu'
    print("Using:", device)
    nettcr = NetTCR(9, 30, 30, device=device)
    print(nettcr.layers)