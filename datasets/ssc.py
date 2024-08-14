import math
from typing import Optional

import torch
import torch.utils.data

from torch.utils.data import DataLoader
import pytorch_lightning as pl
from datasets.utils.pad_tensors import PadTensors
from datasets.utils.transforms import (
    TakeEventByTime,
    Flatten,
)
import tonic
from tonic.transforms import ToFrame


class SSCLDM(pl.LightningDataModule):
    def __init__(
        self,
        data_path: str,
        spatial_factor: float = 0.2,
        time_factor: float = 1e-3,
        window_size: float = 4.0,
        batch_size: int = 256,
        num_workers: int = 1,
        pad_to_min_size: int = 300,
        num_classes: int = 35,
        ignore_first_timesteps: int = 10,
    ) -> None:
        super().__init__()
        # workaround in order to use the same training loop
        # for classification and context processing
        self.data_path = data_path
        self.spatial_factor = spatial_factor
        self.time_factor = time_factor
        self.window_size = window_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.collate_fn = PadTensors()
        self.min_len = pad_to_min_size
        self.ignore_first_timesteps = ignore_first_timesteps

        sensor_size = tonic.datasets.SSC.sensor_size
        sensor_size = (
            int(math.ceil(sensor_size[0] * self.spatial_factor)),
            int(math.ceil(sensor_size[1]) * self.spatial_factor),
            sensor_size[2],
        )

        self.input_size = math.prod(sensor_size)
        self.output_size = 35
        self.class_weights = torch.ones(
            size=(self.output_size,),
        )
        self.sensor_size = sensor_size

        _event_to_tensor = ToFrame(
            sensor_size=sensor_size, time_window=self.window_size
        )
        event_to_tensor = lambda x: torch.from_numpy(_event_to_tensor(x)).float()

        if self.min_len > 0:
            def pad_to_min_len(x):
                if x.shape[0] < self.min_len:
                    pad = torch.zeros((self.min_len - x.shape[0], 1, x.shape[-1]))
                    x = torch.cat((x, pad), dim=0)
                return x
        else:
            def pad_to_min_len(x):
                return x
        transform_list = [
            tonic.transforms.Downsample(
                time_factor=self.time_factor, spatial_factor=self.spatial_factor
            ),
            event_to_tensor,
            pad_to_min_len,
            Flatten(),
        ]
        self.static_data_transform = tonic.transforms.Compose(transform_list)

    def prepare_data(self):
        pass
       
    def setup(self, stage: Optional[str] = None) -> None:
        self.data_train = SSCWrapper(
            save_to=self.data_path,
            split="train",
            transform=self.static_data_transform,
            ignore_first_timesteps=self.ignore_first_timesteps
        )
        self.data_test = SSCWrapper(
            save_to=self.data_path,
            split="test",
            transform=self.static_data_transform,
            ignore_first_timesteps=self.ignore_first_timesteps
        )
        self.data_val = SSCWrapper(
            save_to=self.data_path,
            split="valid",
            transform=self.static_data_transform,
            ignore_first_timesteps=self.ignore_first_timesteps
        )

    def train_dataloader(self):
        return DataLoader(
            self.data_train,
            shuffle=True,
            pin_memory=True,
            batch_size=self.batch_size,
            drop_last=True,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self):
        return DataLoader(
            self.data_val,
            shuffle=False,
            pin_memory=True,
            batch_size=self.batch_size,
            drop_last=False,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self):
        return DataLoader(
            self.data_test,
            pin_memory=True,
            shuffle=False,
            batch_size=self.batch_size,
            drop_last=False,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.data_test,
            shuffle=False,
            batch_size=self.batch_size,
            pin_memory=True,
            drop_last=False,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )


class SSCWrapper(tonic.datasets.SSC):
    dataset_name = "SSC"

    def __init__(
        self,
        save_to: str,
        split: str = "train",
        transform = None,
        target_transform = None,
        ignore_first_timesteps: int = 10,
    ):
        super().__init__(save_to, split, transform, target_transform)
        self.ignore_first_timesteps = ignore_first_timesteps

    def __getitem__(self, index):
        events, target = super().__getitem__(index)
        block_idx = torch.ones((events.shape[0],), dtype=torch.int64)
        block_idx[: self.ignore_first_timesteps] = 0
        return events, target, block_idx
