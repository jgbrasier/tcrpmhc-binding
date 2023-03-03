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

class LightningGCNN(pl.LightningModule):
    """
    Model from:
    Jha, K., Saha, S. & Singh, H. Prediction of protein–protein interaction using graph neural networks. 
    Sci Rep 12, 8360 (2022). https://doi.org/10.1038/s41598-022-12201-9
    """
    def __init__(self, n_output=1, embedding_dim= 1280, output_dim=128, dropout=0.2,
                include_sequence: bool = False, learning_rate = 0.001, device = torch.device('cpu')):
        super().__init__()

        self.save_hyperparameters()

        # for protein 1
        self.pro1_conv1 = GCNConv(self.hparams.embedding_dim, self.hparams.embedding_dim)
        self.pro1_conv2 = GCNConv(self.hparams.embedding_dim, self.hparams.embedding_dim)
        self.pro1_fc1 = nn.Linear(self.hparams.embedding_dim, self.hparams.output_dim)

        # for protein 2
        self.pro2_conv1 = GCNConv(self.hparams.embedding_dim, self.hparams.embedding_dim)
        self.pro2_conv2 = GCNConv(self.hparams.embedding_dim, self.hparams.embedding_dim)
        self.pro2_fc1 = nn.Linear(self.hparams.embedding_dim, self.hparams.output_dim)

        self.relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(self.hparams.dropout)
        self.sigmoid = nn.Sigmoid()

        # combined layers
        self.fc1 = nn.Linear(2 * self.hparams.output_dim, 256)
        self.fc2 = nn.Linear(256 ,64)
        self.out = nn.Linear(64, self.hparams.n_output)

        self.loss_fn = nn.BCELoss()
        self._acc = BinaryAccuracy()
        self._precision = BinaryPrecision()
        self._recall = BinaryRecall()
        self._f1 = BinaryF1Score()
        self._auroc = BinaryAUROC()

    def forward(self, pro1_data, pro2_data):
        #get graph input for protein 1 
        pro1_x, pro1_edge_index, pro1_batch = pro1_data.x.type(torch.float), pro1_data.edge_index.type(torch.int64), pro1_data.batch
        # get graph input for protein 2
        pro2_x, pro2_edge_index, pro2_batch = pro2_data.x.type(torch.float), pro2_data.edge_index.type(torch.int64), pro2_data.batch

        x = self.pro1_conv1(pro1_x, pro1_edge_index)
        x = self.relu(x)    
        # x = self.pro1_conv2(pro1_x, pro1_edge_index)
        # x = self.relu(x)

	    # global pooling
        x = global_mean_pool(x, pro1_batch)   

        # flatten
        x = self.relu(self.pro1_fc1(x))
        x = self.dropout(x)

        xt = self.pro2_conv1(pro2_x, pro2_edge_index)
        xt = self.relu(xt)
        # xt = self.pro2_conv2(pro2_x, pro2_edge_index)
        # xt = self.relu(xt)

	    # global pooling
        xt = global_mean_pool(xt, pro2_batch)  

        # flatten
        xt = self.relu(self.pro2_fc1(xt))
        xt = self.dropout(xt)

	    # Concatenation  
        xc = torch.cat((x, xt), 1)

        # add some dense layers
        xc = self.fc1(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        xc = self.fc2(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        out = self.out(xc)
        out = self.sigmoid(out)
        return out

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

    def training_step(self, batch, batch_idx):
        prot1, prot2, label = batch
        label = label.type(torch.float) # output is float32 needs to match
        output = self(prot1, prot2)
        loss = self.loss_fn(output, label)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=False, logger=True, batch_size=len(batch))
        return loss

    def validation_step(self, batch, batch_idx):
        prot1, prot2, label = batch
        output = self(prot1, prot2)
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
        prot1, prot2, label = batch
        output = self(prot1, prot2)
        loss = self.loss_fn(output, label)
        acc = self._acc(output, label)
        precision = self._precision(output, label)
        recall = self._recall(output, label)
        f1 = self._f1(output, label)
        auroc = self._auroc(output, label)
        self.log("test_acc", acc, on_step=True, on_epoch=True, prog_bar=False, logger=True, batch_size=len(batch))
        self.log("test_precision", precision, on_step=True, on_epoch=True, prog_bar=False, logger=True, batch_size=len(batch))
        self.log("test_recall", recall, on_step=True, on_epoch=True, prog_bar=False, logger=True, batch_size=len(batch))
        self.log("test_f1", f1, on_step=True, on_epoch=True, prog_bar=False, logger=True, batch_size=len(batch))
        self.log("test_auroc", auroc, on_step=True, on_epoch=True, prog_bar=False, logger=True, batch_size=len(batch))
        self.log("test_loss", loss, on_step=True, on_epoch=True, prog_bar=False, logger=True, batch_size=len(batch))
        return loss


class GCNN(nn.Module):
    def __init__(self, n_output=1, embedding_dim= 1024, output_dim=128, dropout=0.2):
        super(GCNN, self).__init__()
        print('GCNN Loaded')

        # for protein 1
        self.n_output = n_output
        self.pro1_conv1 = GCNConv(embedding_dim, embedding_dim)
        self.pro1_fc1 = nn.Linear(embedding_dim, output_dim)

        # for protein 2
        self.pro2_conv1 = GCNConv(embedding_dim, embedding_dim)
        self.pro2_fc1 = nn.Linear(embedding_dim, output_dim)

        self.relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()

        # combined layers
        self.fc1 = nn.Linear(2 * output_dim, 256)
        self.fc2 = nn.Linear(256 ,64)
        self.out = nn.Linear(64, self.n_output)

    def forward(self, pro1_data, pro2_data):
        #get graph input for protein 1 
        pro1_x, pro1_edge_index, pro1_batch = pro1_data.x, pro1_data.edge_index, pro1_data.batch
        # get graph input for protein 2
        pro2_x, pro2_edge_index, pro2_batch = pro2_data.x, pro2_data.edge_index, pro2_data.batch

        x = self.pro1_conv1(pro1_x, pro1_edge_index)
        x = self.relu(x)
        
	    # global pooling
        x = global_mean_pool(x, pro1_batch)   

        # flatten
        x = self.relu(self.pro1_fc1(x))
        # x = self.dropout(x)

        xt = self.pro2_conv1(pro2_x, pro2_edge_index)
        xt = self.relu(xt)

	    # global pooling
        xt = global_mean_pool(xt, pro2_batch)  

        # flatten
        xt = self.relu(self.pro2_fc1(xt))
        # xt = self.dropout(xt)

	    # Concatenation  
        xc = torch.cat((x, xt), 1)

        # add some dense layers
        xc = self.fc1(xc)
        xc = self.relu(xc)
        # xc = self.dropout(xc)
        xc = self.fc2(xc)
        xc = self.relu(xc)
        # xc = self.dropout(xc)
        out = self.out(xc)
        out = self.sigmoid(out)
        return out
        
"""# GAT"""

class AttGNN(nn.Module):
    def __init__(self, n_output=1, embedding_dim= 1024, output_dim=128, dropout=0.2, heads = 1 ):
        super(AttGNN, self).__init__()

        print('AttGNN Loaded')

        self.hidden = 8
        self.heads = heads
        
        # for protein 1
        self.pro1_conv1 = GATConv(embedding_dim, self.hidden* 16, heads=self.heads, dropout=0.2)
        self.pro1_fc1 = nn.Linear(128, output_dim)

        # for protein 2
        self.pro2_conv1 = GATConv(embedding_dim, self.hidden*16, heads=self.heads, dropout=0.2)
        self.pro2_fc1 = nn.Linear(128, output_dim)

        self.relu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout)
        # combined layers
        self.fc1 = nn.Linear(2 * output_dim, 256)
        self.fc2 = nn.Linear(256, 64)
        self.out = nn.Linear(64, n_output)
        


    def forward(self, pro1_data, pro2_data):

        # get graph input for protein 1 
        pro1_x, pro1_edge_index, pro1_batch = pro1_data.x, pro1_data.edge_index, pro1_data.batch
        # get graph input for protein 2
        pro2_x, pro2_edge_index, pro2_batch = pro2_data.x, pro2_data.edge_index, pro2_data.batch
         
        x = self.pro1_conv1(pro1_x, pro1_edge_index)
        x = self.relu(x)
        
	    # global pooling
        x = global_mean_pool(x, pro1_batch)  
       
        # flatten
        x = self.relu(self.pro1_fc1(x))
        x = self.dropout(x)

        xt = self.pro2_conv1(pro2_x, pro2_edge_index)
        xt = self.relu(self.pro2_fc1(xt))
	
	    # global pooling
        xt = global_mean_pool(xt, pro2_batch)  

        # flatten
        xt = self.relu(xt)
        xt = self.dropout(xt)
	
	    # Concatenation
        xc = torch.cat((x, xt), 1)

        # add some dense layers
        xc = self.fc1(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        xc = self.fc2(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        out = self.out(xc)
        out = self.sigmoid(out)
        return out


if __name__=="__main__":
    net = GCNN()
    print(net)
    net_GAT = AttGNN()
    print(net_GAT)  
