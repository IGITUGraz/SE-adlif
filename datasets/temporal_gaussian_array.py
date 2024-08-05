from typing import Optional
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
import numpy as np
from datasets.utils.pad_tensors import PadTensors
from functional.encode import poisson_encode
import torch
from torch.nn.functional import pad as torch_pad
import random

class TemporalGaussianArray(Dataset):
    dataset_name = "TemporalGaussianArray"
    def __init__(
        self,
        sampling_freq: int,
        sample_len: int,
        min_num_sample: int,
        max_num_sample: int,
        delay_between_sample: int,
        num_sample_per_epoch: int,
        transform = None,
        target_transform = None,
        class_parameters=None,
        exp_cfg=None
    ):
        super().__init__()
        self.transform = transform
        self.target_transform = target_transform
        self.min_num_sample = min_num_sample
        self.max_num_sample = max_num_sample
        self.sample_len = sample_len
        self.sampling_freq = sampling_freq
        self.num_sample_per_epoch = num_sample_per_epoch
        self.delay_between_sample = delay_between_sample
        self.class_parameters = class_parameters
        self.exp_cfg = exp_cfg
        self.S = exp_cfg["S"]  # Length of each array
        self.n_channels = exp_cfg["n_channels"]  # Total channels per sample
        self.n_classes = exp_cfg["n_classes"]  # Number of distinct classes
        self.n_common_channels = exp_cfg["n_common_channels"]  # Common channels per class
        self.n_spikes = exp_cfg["n_spikes"]  # Gaussian spikes per sample
        self.n_noise_spikes = exp_cfg["n_noise_spikes"]  # Noise spikes per sample
        
    def __getitem__(self, index):

        # Get an example multidigit addition:
        '''
        n_digits=self.n_digits # former num_samples
        one_hot_size = 20

        inputs, targets = generate_dataset(n_samples=1, n_digits=n_digits, input_representation='one_hot', output_representation='numerical')
        inputs = torch.from_numpy(inputs[0])
        targets = torch.from_numpy(targets[0])
        '''

        ### Temporal gaussian array:

        class_parameters = self.class_parameters
        S = self.S
        n_classes = self.n_classes
        n_channels = self.n_channels
        n_common_channels = self.n_common_channels
        n_spikes = self.n_spikes
        n_noise_spikes = self.n_noise_spikes

        class_id = np.random.randint(n_classes)  # Assign class in a round-robin fashion
        class_means, class_variances, common_channels = class_parameters[class_id]

        means, variances = generate_sample_parameters(n_channels, class_means, class_variances, n_common_channels,
                                                      common_channels)
        sample = create_sample(S, n_channels, means, variances, n_spikes, n_noise_spikes)
        inputs = torch.tensor(sample, dtype=torch.float32)
        targets = torch.tensor(class_id, dtype=torch.int32)


        ### End Temporal gaussian array
        inputs = inputs.transpose(0,1)

        # create a indexing vector for each target starting
        # from the index 1, index 0 is defined for the padding
        block_idx = torch.zeros(S, dtype=torch.int32)
        block_idx[int(S*0.8):] = 1

        if self.transform is not None:
            inputs = self.transform(inputs)
        if self.target_transform is not None:
            targets = self.target_transform(targets)
        
        return inputs, targets.unsqueeze(0), block_idx
        
    def __len__(self):
        return self.num_sample_per_epoch


class TemporalGaussianArrayDM(pl.LightningDataModule):

    def __init__(self,
                batch_size: int,
                num_workers: int,
                sampling_freq: int,
                sample_len: int,
                min_num_sample: int,
                max_num_sample: int,
                delay_between_sample: int,
                num_sample_per_epoch: int,
                S: int,
                n_channels: int,
                n_classes: int,
                n_common_channels: int,
                n_spikes: int,
                n_noise_spikes: int,
                val_split: 0.1,
                test_split: 0.2,
                name: str = None,  # for hydra

    ) -> None:
        super().__init__()

        self.collate_fn = PadTensors()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split
        self.test_split = test_split
        #self.n_digits = n_digits

        exp_cfg = {}
        exp_cfg["S"] = S
        exp_cfg["n_channels"] = n_channels
        exp_cfg["n_classes"] = n_classes
        exp_cfg["n_common_channels"] = n_common_channels
        exp_cfg["n_spikes"] = n_spikes
        exp_cfg["n_noise_spikes"] = n_noise_spikes


        self.class_parameters = initialize_class_parameters(n_classes, n_channels, n_common_channels)
        self.gen_dataset = TemporalGaussianArray(
            sampling_freq=sampling_freq,
            sample_len=sample_len,
            min_num_sample=min_num_sample,
            max_num_sample=max_num_sample,
            delay_between_sample=delay_between_sample,
            num_sample_per_epoch=num_sample_per_epoch,
            class_parameters=self.class_parameters,
            exp_cfg=exp_cfg
        )
        
    def prepare_data(self):
        generator1 = torch.Generator().manual_seed(42)
        #self.train_dataset_, self.valid_dataset_ , self.test_dataset_ = torch.utils.data.random_split(self.gen_dataset, (1.0-self.val_split-self.test_split, self.val_split, self.test_split), generator=generator1)
        self.test_dataset_ = self.gen_dataset
        self.train_dataset_ = self.gen_dataset
        self.valid_dataset_ = self.gen_dataset

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == "fit" or stage == "validate":
            self.data_train = self.train_dataset_
            self.data_val = self.valid_dataset_
        if stage == "test" or stage == "predict":
            self.data_test = self.test_dataset_

    def train_dataloader(self):
        return DataLoader(self.data_train, 
                          shuffle=True,
                          pin_memory=True,
                          batch_size=self.batch_size,
                          drop_last=True,
                          collate_fn=self.collate_fn,
                          num_workers=self.num_workers,
                          persistent_workers=self.num_workers > 0,)

    def val_dataloader(self):
        return DataLoader(self.data_val,
                          shuffle=False,
                          pin_memory=True,
                          batch_size=self.batch_size,
                          drop_last=False,
                          collate_fn=self.collate_fn,
                          num_workers=self.num_workers,
                          persistent_workers=self.num_workers > 0)

    def test_dataloader(self):
        return DataLoader(self.data_test,
                          pin_memory=True,
                          shuffle=False,
                          batch_size=self.batch_size,
                          drop_last=False,
                          collate_fn=self.collate_fn,
                          num_workers=self.num_workers,
                          persistent_workers=self.num_workers > 0)

    def predict_dataloader(self):
        return DataLoader(self.data_test,
                          shuffle=False,
                          batch_size=self.batch_size,
                          pin_memory=True,
                          drop_last=False,
                          collate_fn=self.collate_fn,
                          num_workers=self.num_workers,
                          persistent_workers=self.num_workers > 0)


import numpy as np


###############################################
### Temporal gaussian array
###############################################

def create_gaussian_array(S, mean, variance, n_spikes):
    # Adjust mean to scale with array length
    adjusted_mean = mean * S

    # Compute variance as a random value between var_min and var_max
    # variance = np.random.uniform(var_min, var_max)

    # Generate Gaussian distributed indices
    gaussian_indices = np.random.normal(loc=adjusted_mean, scale=np.sqrt(variance), size=n_spikes)

    # Initialize the binary array
    binary_array = np.zeros(S, dtype=int)

    # Convert indices to valid array positions (ensuring they fall within the array bounds)
    valid_indices = np.clip(gaussian_indices, 0, S - 1).astype(int)

    # Set the positions to 1 according to Gaussian indices
    np.add.at(binary_array, valid_indices, 1)

    return binary_array


def create_sample(S, n_channels, means, variances, n_spikes, n_noise_spikes):
    # Initialize the sample with multiple channels
    sample = np.zeros((n_channels, S), dtype=int)

    for i in range(n_channels):
        # Generate the Gaussian-distributed array for each channel with individualized parameters
        sample[i] = create_gaussian_array(S, means[i], variances[i], n_spikes)

        # Add noise spikes randomly
        noise_indices = np.random.randint(0, S, n_noise_spikes)
        np.add.at(sample[i], noise_indices, 1)

    return sample


def initialize_class_parameters(n_classes, n_channels, n_common_channels):
    class_params = []
    for _ in range(n_classes):
        # Generate distinct parameters for n_common_channels
        means = np.random.rand(n_common_channels) * 0.8
        variances = np.random.uniform(1, 5, n_common_channels)
        common_channels = random.sample(range(0, n_channels), n_common_channels)
        class_params.append((means, variances, common_channels))
    return class_params


def generate_sample_parameters(n_channels, class_means, class_variances, n_common_channels, common_channels):
    means = np.random.rand(n_channels) * 0.8
    variances = np.random.uniform(1, 5, n_channels)

    # Override common_channels with class-specific parameters
    means[common_channels] = class_means
    variances[common_channels] = class_variances

    return means, variances


def create_dataset(S, n_samples, n_channels, n_classes, n_common_channels, n_spikes, n_noise_spikes):
    class_parameters = initialize_class_parameters(n_classes, n_channels, n_common_channels)
    dataset = []
    labels = []

    for i in range(n_samples):
        class_id = i % n_classes  # Assign class in a round-robin fashion
        class_means, class_variances, common_channels = class_parameters[class_id]

        means, variances = generate_sample_parameters(n_channels, class_means, class_variances, n_common_channels,
                                                      common_channels)
        sample = create_sample(S, n_channels, means, variances, n_spikes, n_noise_spikes)

        dataset.append(sample)
        labels.append(class_id)

    return np.array(dataset), np.array(labels), class_parameters

