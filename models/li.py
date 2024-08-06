from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module
from torch import Tensor
from torch.nn.parameter import Parameter

from module.tau_trainers import TauTrainer, get_tau_trainer_class
# TODO: remove this comment
# what was removed:
# any tracking hook for plotting
# initialization method: as we use the same things every time
# use bias: as we always use bias
# what was kept:
# tau_mapping: nice encapsulation and 
# we sometime train the tau of the last layer (auto-regression task)
from omegaconf import DictConfig

class LI(Module):
    __constants__ = ["in_features", "out_features"]
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(
        self,
        cfg: DictConfig,
        device=None,
        dtype=None,
        **kwargs,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__(**kwargs)
        self.in_features = cfg.n_neurons
        self.out_features = cfg.dataset.num_classes
        self.dt = 1.0
        self.tau_u_range = cfg.tau_out_range
        self.train_tau_u_method = cfg.get('train_tau_out_method', 'fixed')
        self.weight = Parameter(
            torch.empty((self.out_features, self.in_features), **factory_kwargs)
        )

        self.bias = Parameter(torch.empty(self.out_features, **factory_kwargs))
        self.tau_u_trainer: TauTrainer = get_tau_trainer_class(self.train_tau_u_method)(
            self.out_features,
            self.dt,
            self.tau_u_range[0],
            self.tau_u_range[1],
            **factory_kwargs,
        )

        self.reset_parameters()

    def reset_parameters(self):
        self.tau_u_trainer.reset_parameters()
        nn.init.uniform_(
            self.weight,
            -1 * torch.sqrt(1 / torch.tensor(self.in_features)),
            torch.sqrt(1 / torch.tensor(self.in_features)),
        )
        torch.nn.init.zeros_(self.bias)

    @torch.jit.ignore
    def extra_repr(self) -> str:
        return "in_features={}, out_features={}, bias={}".format(
            self.in_features, self.out_features, self.bias is not None
        )

    def initial_state(
        self, batch_size: int, device: Optional[torch.device] = None
    ) -> Tensor:
        size = (batch_size, self.out_features)
        u = torch.zeros(size=size, device=device, dtype=torch.float, requires_grad=True)
        return u

    def forward(self, input_tensor: Tensor, states: Tensor) -> Tuple[Tensor, Tensor]:
        u_tm1 = states
        decay_u = self.tau_u_trainer.get_decay()
        current = F.linear(input_tensor, self.weight, self.bias)
        u_t = decay_u * u_tm1 + (1.0 - decay_u) * current
        return u_t, u_t

    def apply_parameter_constraints(self):
        self.tau_u_trainer.apply_parameter_constraints()