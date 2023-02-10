import os
import pandas as pd
import numpy as np

from src.utils import hard_split_df

import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

class ImageClassificationDataset(Dataset):
    def __init__(self, df: pd.DataFrame, dist_mat_dir: str, id_col: str ='uuid', y_col: str ='binder',):
        self.dist_mat_dir = dist_mat_dir
        self.df = df

        self._id_col = id_col
        self._y_col = y_col

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        pdb_id = str(self.labels_df.iloc[idx][self._id_col])
        img_path = os.path.join(self.dist_mat_dir, pdb_id+'.npy')
        image = torch.from_numpy(np.load(img_path))
        label = self.df.iloc[idx][self._y_col]
        return (image, label)

class ImageClassificationDataModule(pl.LightningDataModule):
    """
    """
    def __init__(self, tsv_path: str = None, processed_dir: str = None, id_col: str ='uuid', y_col='binder', \
                batch_size: int = 32, num_workers: int = 0, device=torch.device('cpu')):
        super().__init__()
        self.save_hyperparameters()

        self.df = pd.read_csv(tsv_path, sep='\t')
        # self.df = pd.concat((self.df[self.df[y_col]==0], self.df[self.df[y_col]==1].sample(frac=0.2, random_state=1)))

        self.train: ImageClassificationDataset = None
        self.val: ImageClassificationDataset = None
        self.test: ImageClassificationDataset = None

        self.selected_targets = None

    def setup(self, train_size: int = 0.8, target='epitope', low: int = 50, high: int = 800, random_seed: int = 42):
        assert train_size > 0 and train_size <= 1, "train_size must be in (0, 1]"
        train_df, test_df, self.selected_targets = hard_split_df(self.df, target_col=target, min_ratio=train_size,
                                                    low=low, high=high, random_seed=random_seed)

        self.train = ImageClassificationDataset(train_df, self.hparams.processed_dir, self.hparams.id_col, self.hparams.y_col, device=self.hparams.device)
        if train_size == 1:
            self.test = None
        else:
            self.test = ImageClassificationDataset(test_df, self.hparams.processed_dir, self.hparams.id_col,self.hparams.y_col, device=self.hparams.device)
        return self.selected_targets

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers, shuffle=True)  # type: ignore

    def val_dataloader(self):
        # TODO:
        raise NotImplementedError
        return DataLoader(self.val, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers, shuffle=True)  # type: ignore
    
    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers, shuffle=True)  # type: ignore