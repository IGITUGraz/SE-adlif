import math
from typing import Optional

import numpy as np
import torch
import torch.utils.data
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from datasets.utils.pad_tensors import PadTensors
from datasets.utils.transforms import TakeEventByTime, AddNoiseByproportion, Flatten
import tonic
from tonic.datasets.hsd import SHD
from torch.utils import data
from tonic.transforms import ToFrame


""" The following class is a wrapper for the SHD dataset with a block_idx (see Readme.md) """


class SHDWrapper(SHD):
    dataset_name = "SHD"

    def __init__(
        self,
        save_to: str,
        train: bool = True,
        transform = None,
        target_transform = None,
        ignore_first_timesteps: int = 10,
    ):
        super().__init__(save_to, train, transform, target_transform)
        self.ignore_first_timesteps = ignore_first_timesteps

    def __getitem__(self, index):
        events, target = super().__getitem__(index)
        block_idx = torch.ones((events.shape[0],), dtype=torch.int64)
        block_idx[:self.ignore_first_timesteps] = 0
        return events, target, block_idx


class SHDLDM(pl.LightningDataModule):
    def __init__(
        self,
        data_path: str,
        spatial_factor: float = 0.12,
        time_factor: float = 1e-3,
        window_size: float = 5.0,
        split_percent: float = 0.8,
        batch_size: int = 32,
        num_workers: int = 1,
        pad_to_min_size: int = None,
        bias_for_test_set: float = 0.0,
        test_bias_variant: str = "bias",
        name: str = None,  # for hydra
        num_classes: int = 20,  # for hydra
        validate_on=0.05,
        additional_test_set_validation: bool = False,
        random_seed=42,
        ignore_first_timesteps: int = 10,
    ) -> None:
        super().__init__()
        # workaround in order to use the same training loop
        # for classification and context processing
        self.data_path = data_path
        self.spatial_factor = spatial_factor
        self.time_factor = time_factor
        self.window_size = window_size
        self.split_percent = split_percent
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.collate_fn = PadTensors()
        self.validate_on = validate_on
        self.additional_test_set_validation = additional_test_set_validation
        self.random_seed = random_seed
        self.bias_for_test_set = bias_for_test_set
        self.test_bias_variant = test_bias_variant
        # used for the random position signal task
        # the transformation will add zeros at the right side
        # of a length based on the original size of the data-sample
        self.pad_to_min_size = pad_to_min_size
        self.ignore_first_timesteps = ignore_first_timesteps

        sensor_size = SHDWrapper.sensor_size
        sensor_size = (
            int(math.ceil(sensor_size[0] * self.spatial_factor)),
            int(math.ceil(sensor_size[1]) * self.spatial_factor),
            sensor_size[2],
        )

        self.input_size = math.prod(sensor_size)
        self.output_size = 20
        self.class_weights = torch.ones(
            size=(self.output_size,),
        )
        self.sensor_size = sensor_size
        _event_to_tensor = ToFrame(
            sensor_size=sensor_size, time_window=self.window_size
        )

        def event_to_tensor(x):
            return torch.from_numpy(_event_to_tensor(x)).float()

        def pad_to_min_size(x):
            if self.pad_to_min_size is None:
                return x
            elif x.shape[0] >= self.pad_to_min_size:
                return x
            else:
                return torch.cat(
                    [
                        x,
                        torch.zeros(
                            (self.pad_to_min_size - x.shape[0], x.shape[1], x.shape[2])
                        ),
                    ],
                    dim=0,
                )

        def add_bias_to_test_data(x, level):
            return x + level * torch.mean(x)

        if test_bias_variant == "noise":
            test_transform_list = [
                AddNoiseByproportion(
                    self.bias_for_test_set,
                    is_frame=False,
                    sensor_size=(sensor_size[0], 1, 1),
                ),
                tonic.transforms.Downsample(
                    time_factor=self.time_factor, spatial_factor=self.spatial_factor
                ),
                event_to_tensor,
                # pooling,
                pad_to_min_size,
                Flatten(),
            ]
        else:
            test_transform_list = [
                tonic.transforms.Downsample(
                    time_factor=self.time_factor, spatial_factor=self.spatial_factor
                ),
                event_to_tensor,
                # pooling,
                pad_to_min_size,
                Flatten(),
            ]

        self.static_data_transform_train_val = tonic.transforms.Compose(
            [
                tonic.transforms.Downsample(
                    time_factor=self.time_factor, spatial_factor=self.spatial_factor
                ),
                event_to_tensor,
                pad_to_min_size,
                Flatten(),
            ]
        )  # noqa

        # This is used for the noisy experiments
        if test_bias_variant == "bias":
            self.static_data_transform_test = tonic.transforms.Compose(
                test_transform_list
                + [lambda x: add_bias_to_test_data(x, self.bias_for_test_set)]
            )
        else:
            self.static_data_transform_test = tonic.transforms.Compose(
                test_transform_list
            )

        self.generator = torch.Generator().manual_seed(self.random_seed)
        self.data_test = SHDWrapper(
            save_to=self.data_path,
            transform=self.static_data_transform_test,
            train=False,
            ignore_first_timesteps=self.ignore_first_timesteps,
        )

        # validate on the test set
        if self.validate_on == "test":
            self.data_train = SHDWrapper(
                save_to=self.data_path,
                transform=self.static_data_transform_train_val,
                train=True,
                ignore_first_timesteps=self.ignore_first_timesteps,
            )

            self.data_val = SHDWrapper(
                save_to=self.data_path,
                transform=self.static_data_transform_train_val,
                train=False,
                ignore_first_timesteps=self.ignore_first_timesteps,
            )

        # validate on a percentage of the training set
        elif isinstance(self.validate_on, float):
            self.train_val_ds = SHDWrapper(
                save_to=self.data_path,
                transform=self.static_data_transform_train_val,
                train=True,
                ignore_first_timesteps=self.ignore_first_timesteps,
            )
            valid_len = math.floor(len(self.train_val_ds) * self.validate_on)
            self.data_train, self.data_val = torch.utils.data.random_split(
                self.train_val_ds,
                [len(self.train_val_ds) - valid_len, valid_len],
                generator=self.generator,
            )
        else:
            raise ValueError(f"validate_on {self.validate_on} is not valid")

    def prepare_data(self):
        pass
        

    def setup(self, stage: Optional[str] = None) -> None:
        pass

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
        val_dataloader = DataLoader(
            self.data_val,
            shuffle=False,
            pin_memory=True,
            batch_size=self.batch_size,
            drop_last=False,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )

        if self.additional_test_set_validation:
            return [
                val_dataloader,
                DataLoader(
                    self.data_test,
                    shuffle=False,
                    pin_memory=True,
                    batch_size=self.batch_size,
                    drop_last=False,
                    collate_fn=self.collate_fn,
                    num_workers=self.num_workers,
                    persistent_workers=self.num_workers > 0,
                ),
            ]
        else:
            return val_dataloader

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