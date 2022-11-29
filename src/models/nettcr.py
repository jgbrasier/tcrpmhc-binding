import numpy as np
import torch

import torch.nn as nn
from torch.nn import Linear, Conv1d, AdaptiveMaxPool1d

import pytorch_lightning as pl


class NetTCR(pl.LightningModule):
    def __init__(self, peptide_len: int, cdrb_len: int, cdra_len: int = None, n_layers: int = 5, batch_size=16, device='cpu'):
        if cdra_len is not None:
            elems = {'peptide': peptide_len, 'cdra': cdra_len, 'cdrb': cdrb_len}
        else:
            elems = {'peptide': peptide_len, 'cdrb': cdrb_len}
        self.batch_size = batch_size

        self._modules = dict()
        for e, l in elems.items():
            self._modules[e] = nn.Sequential()
            for i in range(n_layers):
                self._modules[e].add_module(f"{e}_conv_{i+1}", Conv1d(in_channels=l, out_channels=16, kernel_size=2*i+1, padding='same', device=device))
                self._modules[e].add_module(f"{e}_maxpool_{i+1}",AdaptiveMaxPool1d(output_size=16))
        

if __name__=='__main__':
    device = torch.device('mps') if torch.backends.mps.is_available() else 'cpu'

    nettcr = NetTCR(9, 30, 30)
    print(nettcr._modules)