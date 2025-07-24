from pathlib import Path
from typing import Optional

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.data import Subset
import torch

from .av2_dataset import Av2Dataset, collate_fn


class Av2DataModule(LightningDataModule):
    def __init__(
        self,
        data_root: str,
        data_folder: str,
        train_batch_size: int = 32,
        val_batch_size: int = 32,
        test_batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 8,
        pin_memory: bool = True,
        test: bool = False,
        prefetch_factor: int = 4,
    ):
        super(Av2DataModule, self).__init__()
        self.data_root = Path(data_root)
        self.data_folder = data_folder
        self.batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.test = test
        self.prefetch_factor = prefetch_factor
        

    def setup(self, stage: Optional[str] = None, train_fraction: float = 1.0, 
              val_fraction: float = 1.0, test_fraction: float = 1.0) -> None:
        if not self.test:
            self.full_train_dataset = Av2Dataset(
                data_root=self.data_root / self.data_folder, cached_split="train"
            )
            self.full_val_dataset = Av2Dataset(
                data_root=self.data_root / self.data_folder, cached_split="val"
            )
            
            if train_fraction < 1.0:
                train_size = int(len(self.full_train_dataset) * train_fraction)
                val_size = int(len(self.full_val_dataset) * val_fraction)
                train_indices = torch.randperm(len(self.full_train_dataset))[:train_size]
                val_indices = torch.randperm(len(self.full_val_dataset))[:val_size]
                self.train_dataset = Subset(self.full_train_dataset, train_indices)
                self.val_dataset = Subset(self.full_val_dataset, val_indices)
                print(f"Using {train_fraction*100}% of the training and validation datasets.")
            else:
                self.train_dataset = self.full_train_dataset
                self.val_dataset = self.full_val_dataset
                print("Using the full training and validation datasets.")
        else:
            self.full_test_dataset = Av2Dataset(
                data_root=self.data_root / self.data_folder, cached_split="test"
            )
            
            if test_fraction < 1.0:
                test_size = int(len(self.full_test_dataset) * test_fraction)
                test_indices = torch.randperm(len(self.full_test_dataset))[:test_size]
                self.test_dataset = Subset(self.full_test_dataset, test_indices)
                print(f"Using {test_fraction*100}% of the test dataset.")
            else:
                self.test_dataset = self.full_test_dataset
                print("Using the full test dataset.")

    def train_dataloader(self):
        return TorchDataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            pin_memory=self.pin_memory,
            collate_fn=collate_fn,
        )

    def val_dataloader(self):
        return TorchDataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=collate_fn,
        )

    def test_dataloader(self):
        return TorchDataLoader(
            self.test_dataset,
            batch_size=self.test_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=collate_fn,
        )