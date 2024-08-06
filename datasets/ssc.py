from copy import deepcopy
from logging import Logger
import math
import os
from typing import Optional, Union
import debugpy

import numpy as np
import numpy.lib.recfunctions
import torch
import torch.utils.data
from traitlets import Callable
from datasets.cached_dataset import DiskCachedDataset

# from tonic import DiskCachedDataset
from torch.utils.data import DataLoader
from datasets.dataset import Dataset
import pytorch_lightning as pl
from pathlib import Path
from datasets.utils.pad_tensors import PadTensors
from functional.transforms import (
    PhaseContextTransform,
    PoissonContextTransform,
    TakeEventByTime,
    ToFrame,
    BinarizeFrame,
    Denoise1D,
)
from functional.utils import Flatten
import tonic
from tonic.datasets.hsd import SHD
import h5py
from tonic.io import make_structured_array
from tonic.transforms import CropTime
from torch.utils import data
from torch.nn import AvgPool2d, AvgPool1d


class SSCLDM(pl.LightningDataModule):
    def __init__(
        self,
        data_path: str,
        cache_root: str = None,
        spatial_factor: float = 0.12,
        time_factor: float = 1e-3,
        input_encoding: str = "time_window",
        n_time_bins: int = 1000,
        window_size: float = 5.0,
        count_thr: int = 3,
        duration_ratio: float = 0.4,
        split_percent: float = 0.8,
        batch_size: int = 32,
        get_metadata: bool = False,
        num_workers: int = 1,
        fits_into_ram: bool = False,
        min_len: int = -1,
        name: str = None,  # for hydra
        required_model_size: str = None,  # for hydra
        num_classes: int = 35,
    ) -> None:
        super().__init__()
        # workaround in order to use the same training loop
        # for classification and context processing
        self.context_size = 1
        self.data_path = data_path
        self.spatial_factor = spatial_factor
        self.time_factor = time_factor
        self.input_encoding = input_encoding
        self.n_time_bins = n_time_bins
        self.window_size = window_size
        self.count_thr = count_thr
        self.duration_ratio = duration_ratio
        self.split_percent = split_percent
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.fits_into_ram = fits_into_ram
        self.get_metadata = get_metadata
        self.collate_fn = PadTensors(sparse_data=False, contextual=False)
        self.cache_root = data_path if cache_root == None else cache_root
        self.min_len = min_len

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

        if self.input_encoding == "time_bin":
            _event_to_tensor = ToFrame(
                sensor_size=sensor_size, n_time_bins=self.n_time_bins
            )  # noqa
            event_to_tensor = lambda x: torch.from_numpy(_event_to_tensor(x)).float()
        elif self.input_encoding == "time_window":
            # event_to_tensor = BinarizeFrame(
            #     sensor_size=sensor_size, time_window=self.window_size,
            #     count_thr=self.count_thr, sparsify=False)
            _event_to_tensor = ToFrame(
                sensor_size=sensor_size, time_window=self.window_size
            )
            event_to_tensor = lambda x: torch.from_numpy(_event_to_tensor(x)).float()
        else:
            raise NotImplementedError(
                f"Input encoding {self.input_encoding} does not exist"
            )
        if min_len > 0:
            def pad_to_min_len(x):
                if x.shape[0] < self.min_len:
                    pad = torch.zeros((self.min_len - x.shape[0], 1, x.shape[-1]))
                    x = torch.cat((x, pad), dim=0)
                return x
        else:
            def pad_to_min_len(x):
                return x
        transform_list = [
            TakeEventByTime(self.duration_ratio),
            tonic.transforms.Downsample(
                time_factor=self.time_factor, spatial_factor=self.spatial_factor
            ),
            event_to_tensor,
            pad_to_min_len,
            Flatten(),
        ]
        self.static_data_transform = tonic.transforms.Compose(transform_list)  # noqa
        self.cache_data_transform = None

        cache_path = (
            f"{self.cache_root}/cache/ssc_classif/"
            f"_{self.spatial_factor}_{self.time_factor}_{self.input_encoding}"
            f"_{self.n_time_bins}_{self.window_size}_{self.count_thr}"
            f"_{self.duration_ratio}_{self.get_metadata}_{self.min_len}_new_block_idx"
        )
        self.cache_path = cache_path
        print("Data path: {}, Cache path: {}".format(self.data_path, self.cache_path))

        # self.save_hyperparameters()

    def prepare_data(self):
        self.train_dataset_ = SSCWrapper(
            save_to=self.data_path,
            split="train",
            transform=self.static_data_transform,
            get_metadata=self.get_metadata,
        )
        self.test_dataset_ = SSCWrapper(
            save_to=self.data_path,
            split="test",
            transform=self.static_data_transform,
            get_metadata=self.get_metadata,
        )
        self.valid_dataset_ = SSCWrapper(
            save_to=self.data_path,
            split="valid",
            transform=self.static_data_transform,
            get_metadata=self.get_metadata,
        )

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == "validate":
            self.data_val = DiskCachedDataset(
                self.valid_dataset_,
                contextual=False,
                cache_path=self.cache_path + "/val",
                transform=self.cache_data_transform,
            )
        if stage == "fit":
            full_data = self.train_dataset_
            # data_train, data_val = split_dataset(
            #     full_data, self.split_percent)
            if self.cache_path != "":
                # self.data_train=full_data
                # full_data = SHDWrapper(
                #     save_to=self.data_path,
                #     transform=self.static_data_transform,
                #     train=False,
                # )
                # self.data_val = full_data
                # if self.fits_into_ram:
                #     self.data_train = RamCachedDataset(
                #         full_data,
                #         transform=self.cache_data_transform
                #     )
                # else:
                self.data_train = DiskCachedDataset(
                    full_data,
                    contextual=False,
                    cache_path=self.cache_path + "/train",
                    transform=self.cache_data_transform,
                )

                full_data = self.valid_dataset_
                # if self.fits_into_ram:
                #     self.data_val = RamCachedDataset(
                #         full_data,
                #         transform=self.cache_data_transform
                #     )
                # else:
                self.data_val = DiskCachedDataset(
                    full_data,
                    contextual=False,
                    cache_path=self.cache_path + "/val",
                    transform=self.cache_data_transform,
                )
        if stage == "test" or stage == "predict":
            data_test = self.test_dataset_
            # if self.fits_into_ram:
            #     self.data_test = RamCachedDataset(
            #         data_test,
            #         transform=self.cache_data_transform
            #     )
            # else:
            self.data_test = DiskCachedDataset(
                data_test,
                contextual=False,
                cache_path=self.cache_path + "/test",
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
        transform: Optional[Callable] = None,
        get_metadata: bool = False,
        target_transform: Optional[Callable] = None,
    ):
        super().__init__(save_to, split, transform, target_transform)
        self.get_metadata = get_metadata
        file = h5py.File(
            os.path.join(self.location_on_system, self.data_filename), "r")
  
        if get_metadata:
            self.metadata = []
            self.metadata= [
                (file["extra"]["speaker"][i],) 
                  for i in range(super().__len__())]

    def __getitem__(self, index):
        events, target = super().__getitem__(index)
        block_idx = torch.ones((events.shape[0],), dtype=torch.int64)
        if self.get_metadata:
            return events, target, block_idx, self.metadata[index]
        else:
            return events, target, block_idx
