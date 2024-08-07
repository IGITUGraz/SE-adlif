import math
from typing import Callable, Optional
from torch.utils.data import Dataset
import torch
import numpy as np
import os
from scipy.linalg import eigh
import tonic

import torch.utils.data
from torch.utils.data import DataLoader

import pytorch_lightning as pl

from datasets.utils.pad_tensors import PadTensors


class DampedOscilatorWithFixedParameters(Dataset):
    dataset_name = "DampedOscilator"

    def __init__(
        self,
        n_dof: int,
        mass_coeff_range: np.ndarray,
        spring_coeff_range: np.ndarray,
        damping_coeff_range: np.ndarray,
        max_freq: float,
        duration: float,
        num_sample: float,
        get_metadata: bool = False,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        random_initial_velocity: bool = False,
        mask_first_n_percent: float = 0.0,
    ):
        super().__init__()
        seed = int(os.environ.get("PL_GLOBAL_SEED"))
        rng_gen = np.random.default_rng(seed=seed)
        self.transform = transform
        self.target_transform = target_transform
        self.n_dof = n_dof
        self.get_metadata = get_metadata
        self.mass_coeff = rng_gen.integers(mass_coeff_range[0], mass_coeff_range[1] + 1, size=(self.n_dof,)).astype(float)
        
        self.spring_coeff = rng_gen.integers(spring_coeff_range[0], spring_coeff_range[1] + 1, size=(self.n_dof + 1,)).astype(float)
        self.damping_coeff = rng_gen.integers(damping_coeff_range[0], damping_coeff_range[1] + 1, size=(self.n_dof + 1,)).astype(float)
        self.duration = duration
        self.num_sample = num_sample
        self.max_freq = max_freq
        self.random_initial_velocity = random_initial_velocity
        self.mask_first_n_percent = mask_first_n_percent
        # nyquist frequency, should be strictly more than the max frequency we want to represent
        Fs_optimal = 2.0 * (max_freq + 1)
        # sampling period
        period = 1.0 / Fs_optimal
        self.t_array = np.arange(0, int(duration / period) + 1) * period
        M, K, D = create_damp_oscilator_system(self.mass_coeff, self.spring_coeff, self.damping_coeff)
        eigen_val, eigen_vec = solve_homogeneous_from_second_order(M, K, D)
        self.eigen_val = eigen_val
        self.eigen_vec = eigen_vec
        self.init_positions = rng_gen.normal(loc=0.0, scale=1.0, size=(self.num_sample, len(self.eigen_val)))
        if not self.random_initial_velocity:
            self.init_positions[:, len(self.eigen_val)//2:] = 0.0
    def __getitem__(self, index):
        x, _ = compute_batched_system(self.eigen_val, self.eigen_vec, self.init_positions[index][np.newaxis, ...], self.t_array)
        # get real values
        x = np.real(x[0])
        inputs = x[:-1]
        targets = x[1:]
        mask_idx = math.floor(inputs.shape[0]*self.mask_first_n_percent)
        block_idx = torch.ones((inputs.shape[0],), dtype=torch.int16)
        block_idx[:mask_idx] = int(0)
        
        if self.transform is not None:
            inputs = self.transform(inputs)
        if self.target_transform is not None:
            targets = self.target_transform(targets)
        if self.get_metadata:
            return inputs, targets, block_idx, {}
        else:
            return inputs, targets, block_idx
        
    def __len__(self):
        return self.num_sample

def create_damp_oscilator_system(
    mass_coef: np.ndarray, spring_coef: np.ndarray, damping_coef: np.ndarray
):
    # pre-cond: spring_coef and damping_coef should be equal in size
    n_dof = len(mass_coef)
    n_spring = len(spring_coef)
    n_damping = len(damping_coef)
    if n_spring != n_damping:
        raise AttributeError(
            "The number of springs should be the same as the number of dampers"
        )
    if n_dof >= n_spring:
        raise AttributeError(
            f"not enougth spring for {n_dof}-DOF, if you don't want the system to be right attached add a spring with 0 coefficient"
        )
    M = np.eye(n_dof, n_dof) * mass_coef

    K_ii = np.eye(n_dof, n_dof) * (spring_coef[:-1] + spring_coef[1:])
    D_ii = np.eye(n_dof, n_dof) * (damping_coef[:-1] + damping_coef[1:])
    K_ld = -np.eye(n_dof, n_dof, -1) * spring_coef[:-1][..., np.newaxis]
    K_ud = -np.eye(n_dof, n_dof, 1) * spring_coef[1:][..., np.newaxis]
    D_ld = -np.eye(n_dof, n_dof, -1) * damping_coef[:-1][..., np.newaxis]
    D_ud = -np.eye(n_dof, n_dof, 1) * damping_coef[1:][..., np.newaxis]

    K = K_ld + K_ii + K_ud
    D = D_ld + D_ii + D_ud
    return M, K, D


def solve_homogeneous_from_second_order(M: np.ndarray, K: np.ndarray, D: np.ndarray):
    # solve the system using 1-order reparametrization
    # the system:
    # Mx_2dot + Dx_dot + Kx = 0, where x the solution for n-dimensional space
    # Let
    # v = x_dot
    # v_dot = x_2dot
    # =>
    # I x_dot - I v = 0
    # M v_dot + K x + D v = 0
    # =>
    # A = [I 0]
    #     [0 M]
    # B = [0 -I]
    #     [K  D]
    # A y_dot B y = 0
    # where y(t) = [x(t),v(t)] is a 2*n column vector representing x(t) x_dot(t) (position and velocity)
    # as long as A^-1B is non-singular the system may be solved as y(t) = T^-1 exp(-Dt)T y(0)
    # where y(0) are the initial condition (initial position and velocity) and T columns are the eingenvector of A^{-1}B
    # and D are diagonal matrix of the complex eigenvalues
    # -A^{-1}B  = [0, I]
    #          = [-M^{-1}K, -M^{-1}D]

    # compute A^{-1}B
    # M is the matrix of mass which is diagonal
    m_coef = np.diag(M)
    inv_m_coef = 1.0 / m_coef
    inv_m_K = inv_m_coef[..., np.newaxis] * K
    inv_m_D = inv_m_coef[..., np.newaxis] * D
    inv_A_B = np.block([[np.zeros_like(K), np.eye(len(K))], [-inv_m_K, -inv_m_D]])
    d, T = eigh(K, M)
    (d, T) = np.linalg.eig(inv_A_B)
    return d, T


def compute_batched_system(
    d: np.ndarray, T: np.ndarray, y_0: np.ndarray, t_array: np.ndarray
):
    # (n, 1)
    inv_T = np.linalg.inv(T)

    b = y_0 @ inv_T
    b = np.expand_dims(b, 1)
    t_array = np.expand_dims(t_array, (0, 2))
    # row            col
    # d = (1, n) * (T, 1)
    # (T, n) * ()

    exp_d_t = np.exp(d * t_array)
    exp_d_t_b = exp_d_t * b

    y = exp_d_t_b @ T.T
    n_ddof = len(d) // 2
    x, v = y[..., :n_ddof], y[..., n_ddof:]
    return x, v

class OscilatorTask(pl.LightningDataModule):
    """
    Create data module for the damped oscilator system,
    the parameter of the system are selected randomly 
    but fixed for the trial.
    The objective is to predict the next position.
    The generalization is determined by correct prediction from
    unseen initial conditions.
    max_freq determined the maximum frequency that you want to represent
    sensible value would max_freq > 2*pi*sqrt(max(K)/min(M)), 
    the maximal possible harmonic frequency of one spring-mass system in isolation.
    where K and M are the spring and mass coefficients.
    
    
    """
    def __init__(self,
                 n_dof: int,
                 mass_range: tuple[int, int],
                 spring_range: tuple[int, int],
                 damping_range: tuple[int, int],
                 max_freq: float,
                 duration: float,
                 num_sample: int,
                 batch_size: int = 32,
                 num_workers: int = 1,
                 fits_into_ram: bool = False,
                 name: str = None, # for hydra
                 required_model_size: str=None, # for hydra
                 num_classes: int = 0,
                 random_initial_velocity: bool = False,
                 mask_first_n_percent: float = 0.0,
                 ) -> None:
        super().__init__()
        self.n_dof = n_dof
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.fits_into_ram = fits_into_ram
        self.collate_fn = PadTensors()
        # used for the random position signal task
        # the transformation will add zeros at the right side
        # of a length based on the original size of the data-sample
        
        def to_tensor(x):
            return torch.from_numpy(x).float()
        
        transform_list = [to_tensor,]
        
        self.static_data_transform = tonic.transforms.Compose(
            transform_list)  # noqa
        self.gen_dataset = DampedOscilatorWithFixedParameters(
            n_dof=n_dof,
            mass_coeff_range=mass_range,
            spring_coeff_range=spring_range,
            damping_coeff_range=damping_range,
            max_freq=max_freq,
            duration=duration,
            num_sample=num_sample,
            transform=self.static_data_transform,
            target_transform=self.static_data_transform,
            random_initial_velocity=random_initial_velocity,
            mask_first_n_percent=mask_first_n_percent
        )
        self.train_dataset_, self.valid_dataset_, self.test_dataset_ = torch.utils.data.random_split(
            self.gen_dataset,
            [
                0.8,
                0.1,
                0.1,                   
            ],
            generator=None,
        )
        
    def prepare_data(self):
        pass

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
                          persistent_workers=self.num_workers > 0)

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