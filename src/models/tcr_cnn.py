import torch
import torch.nn as nn
import pytorch_lightning as pl

from torchmetrics.classification.accuracy import BinaryAccuracy
from torchmetrics.classification.auroc import BinaryAUROC
from torchmetrics.classification.precision_recall import BinaryPrecision, BinaryRecall
from torchmetrics.classification.f_beta import BinaryF1Score


class SimpleCNN(nn.Module):
    def __init__(self, input_shape, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=input_shape[0], out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(1438208, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.activation = nn.Sigmoid()


    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.maxpool(x)
        x = self.relu(self.conv3(x))
        x = self.maxpool(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)

        return self.activation(x)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample


    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self,num_blocks, num_classes=1):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv = nn.Conv2d(1, 64, 3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.layer1 = self._make_layer(ResidualBlock, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(ResidualBlock, 128, num_blocks[1], stride=2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)
        self.activation = nn.Sigmoid()

    def _make_layer(self, block, out_channels, num_blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels:  
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels),
            )
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for i in range(1, num_blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.avg_pool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return self.activation(out)

import torchvision.models as models

class ResNetLightningModule(pl.LightningModule):
    def __init__(self, num_classes: int = 1, learning_rate: float = 0.001):
        super().__init__()
        self.save_hyperparameters()

        # init a pretrained resnet
        self.model = ResNet([2, 2])

        # metrics
        self.loss_fn = nn.BCELoss()
        self._acc = BinaryAccuracy()
        self._precision = BinaryPrecision()
        self._recall = BinaryRecall()
        self._f1 = BinaryF1Score()
        self._auroc = BinaryAUROC()

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

    def training_step(self, batch, batch_idx):
        prot, label = batch
        label = torch.unsqueeze(label.type(torch.float), dim=1) # output is float32 needs to match
        output = self(prot)
        loss = self.loss_fn(output, label)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=False, logger=True, batch_size=len(batch))
        return loss

    def validation_step(self, batch, batch_idx):
        prot, label = batch
        output = self(prot)
        label = torch.unsqueeze(label.type(torch.float), dim=1) # output is float32 needs to match
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
        label = torch.unsqueeze(label.type(torch.float), dim=1) # output is float32 needs to match

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





class ResNet50TransferLearning(pl.LightningModule):
    def __init__(self, num_classes: int = 1, learning_rate: float = 0.001):
        super().__init__()
        self.save_hyperparameters()

        # init a pretrained resnet
        backbone = models.resnet50(weights="DEFAULT")
        num_filters = backbone.fc.in_features
        layers = list(backbone.children())[:-1]
        self.feature_extractor = nn.Sequential(*layers)

        # use the pretrained model
        self.fc = nn.Linear(num_filters, num_classes)
        self.activation = nn.Sigmoid()

        # metrics
        self.loss_fn = nn.BCELoss()
        self._acc = BinaryAccuracy()
        self._precision = BinaryPrecision()
        self._recall = BinaryRecall()
        self._f1 = BinaryF1Score()
        self._auroc = BinaryAUROC()

    def forward(self, x):
        self.feature_extractor.eval()
        with torch.no_grad():
            representations = self.feature_extractor(x).flatten(1)
        x = self.fc(representations)
        return self.activation(x)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

    def training_step(self, batch, batch_idx):
        prot, label = batch
        label = torch.unsqueeze(label.type(torch.float), dim=1) # output is float32 needs to match
        output = self(prot)
        loss = self.loss_fn(output, label)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=False, logger=True, batch_size=len(batch))
        return loss

    def validation_step(self, batch, batch_idx):
        prot, label = batch
        output = self(prot)
        label = torch.unsqueeze(label.type(torch.float), dim=1) # output is float32 needs to match
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



