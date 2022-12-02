import numpy as np
import torch

import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, Conv1d, AdaptiveMaxPool1d

# import pytorch_lightning as pl

class GlobalMaxPool1D(nn.Module):
    def __init__(self, data_format='features_last'):
        super(GlobalMaxPool1D, self).__init__()
        self.data_format = data_format
        self.step_axis = 2 if self.data_format == 'features_last' else 1

    def forward(self, input):
        return torch.max(input, axis=self.step_axis).values

class NetTCR(nn.Module):
    def __init__(self, peptide_len: int, cdra_len: int,cdrb_len: int, batch_size=16, device='cpu'):
        super().__init__()

        self._elems = {'peptide': peptide_len, 'cdra': cdra_len, 'cdrb': cdrb_len}
        self.batch_size = batch_size

        self.n_kernels = 5
        n_filters = 16
        hidden_dim = 32

        self.layers = dict()
        self.lin1 = Linear(in_features=3*self.n_kernels*n_filters, out_features=hidden_dim, device=device)
        self.lin2 = Linear(in_features=hidden_dim, out_features=1, device=device)
        self.activation = nn.ReLU()
        for e, l in self._elems.items():
            self.layers[e] = dict()
            for i in range(self.n_kernels):
                self.layers[e][f"conv_{2*i+1}"] = Conv1d(in_channels=l, out_channels=n_filters, kernel_size=2*i+1, padding='same', device=device)
                self.layers[e][f"pool_{2*i+1}"] = GlobalMaxPool1D()
        


    def forward(self, x):
        x_peptide, x_cdra, x_cdrb = x

        # pep_out_list = []
        # for n in range(self.n_kernels):
        #     pep_out_n = self.activation(self.layers['peptide'][f"conv_{2*n+1}"](x_peptide))
        #     pep_out_n = self.layers['peptide'][f"pool_{2*n+1}"](pep_out_n)
        #     pep_out_list.append(pep_out_n)

        # cdra_out_list = []
        # for n in range(self.n_kernels):
        #     cdra_out_n = self.activation(self.layers['cdra'][f"conv_{2*n+1}"](x_cdra))
        #     cdra_out_n = self.layers['cdra'][f"pool_{2*n+1}"](cdra_out_n)
        #     cdra_out_list.append(cdra_out_n)

        # cdrb_out_list = []
        # for n in range(self.n_kernels):
        #     cdrb_out_n = self.activation(self.layers['cdrb'][f"conv_{2*n+1}"](x_cdrb))
        #     cdrb_out_n = self.layers['cdrb'][f"pool_{2*n+1}"](cdrb_out_n)
        #     cdrb_out_list.append(cdrb_out_n)

        # pep_out = torch.cat(pep_out_list, dim=1)
        # cdra_out = torch.cat(cdra_out_list, dim=1)
        # cdrb_out = torch.cat(cdrb_out_list, dim=1)

        pep_out_1 = self.activation(self.layers['peptide']["conv_1"](x_peptide))
        pep_out_1 = self.layers['peptide']["pool_1"](pep_out_1)
        print(pep_out_1)
        pep_out_3 = self.activation(self.layers['peptide']["conv_3"](x_peptide))
        pep_out_3 = self.layers['peptide']["pool_3"](pep_out_3)
        pep_out_5 = self.activation(self.layers['peptide']["conv_5"](x_peptide))
        pep_out_5 = self.layers['peptide']["pool_5"](pep_out_5)
        pep_out_7 = self.activation(self.layers['peptide']["conv_7"](x_peptide))
        pep_out_7 = self.layers['peptide']["pool_7"](pep_out_7)
        pep_out_9 = self.activation(self.layers['peptide']["conv_9"](x_peptide))
        pep_out_9 = self.layers['peptide']["pool_9"](pep_out_9)

        cdra_out_1 = self.activation(self.layers['cdra']["conv_1"](x_cdra))
        cdra_out_1 = self.layers['cdra']["pool_1"](cdra_out_1)
        cdra_out_3 = self.activation(self.layers['cdra']["conv_3"](x_cdra))
        cdra_out_3 = self.layers['cdra']["pool_3"](cdra_out_3)
        cdra_out_5 = self.activation(self.layers['cdra']["conv_5"](x_cdra))
        cdra_out_5 = self.layers['cdra']["pool_5"](cdra_out_5)
        cdra_out_7 = self.activation(self.layers['cdra']["conv_7"](x_cdra))
        cdra_out_7 = self.layers['cdra']["pool_7"](cdra_out_7)
        cdra_out_9 = self.activation(self.layers['cdra']["conv_9"](x_cdra))
        cdra_out_9 = self.layers['cdra']["pool_9"](cdra_out_9)

        cdrb_out_1 = self.activation(self.layers['cdrb']["conv_1"](x_cdrb))
        cdrb_out_1 = self.layers['cdrb']["pool_1"](cdrb_out_1)
        cdrb_out_3 = self.activation(self.layers['cdrb']["conv_3"](x_cdrb))
        cdrb_out_3 = self.layers['cdrb']["pool_3"](cdrb_out_3)
        cdrb_out_5 = self.activation(self.layers['cdrb']["conv_5"](x_cdrb))
        cdrb_out_5 = self.layers['cdrb']["pool_5"](cdrb_out_5)
        cdrb_out_7 = self.activation(self.layers['cdrb']["conv_7"](x_cdrb))
        cdrb_out_7 = self.layers['cdrb']["pool_7"](cdrb_out_7)
        cdrb_out_9 = self.activation(self.layers['cdrb']["conv_9"](x_cdrb))
        cdrb_out_9 = self.layers['cdrb']["pool_9"](cdrb_out_9)

        pep_out = torch.cat([pep_out_1, pep_out_3, pep_out_5, pep_out_7, pep_out_9], dim=1)
        cdra_out = torch.cat([cdra_out_1, cdra_out_3, cdra_out_5, cdra_out_7, cdra_out_9], dim=1)
        cdrb_out = torch.cat([cdrb_out_1, cdrb_out_3, cdrb_out_5, cdrb_out_7, cdrb_out_9], dim=1)

        out = torch.cat([pep_out, cdra_out, cdrb_out], dim=1)

        out = self.activation(self.lin1(out))
        out = self.activation(self.lin2(out))

        return out


if __name__=='__main__':
    device = torch.device('mps') if torch.backends.mps.is_available() else 'cpu'
    print("Using:", device)
    nettcr = NetTCR(9, 30, 30, device=device)
    print(nettcr.layers)