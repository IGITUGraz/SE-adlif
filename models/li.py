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

class LI(Module):
    __constants__ = ["in_features", "out_features"]
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(
        self,
        in_features: int,
        out_features: int,
        tau_u_range: tuple = (20, 20),
        train_tau_u_method: str = "fixed",
        device=None,
        dtype=None,
        **kwargs,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__(**kwargs)
        self.in_features = in_features
        self.out_features = out_features
        self.dt = 1.0
        self.tau_u_range = tau_u_range
        self.weight = Parameter(
            torch.empty((out_features, in_features), **factory_kwargs)
        )

        self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
        self.tau_u_trainer: TauTrainer = get_tau_trainer_class(train_tau_u_method)(
            out_features,
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

    def forward(self, input_tensor: Tensor) -> Tuple[Tensor, Tensor]:
    
        u0 = self.initial_state(
            batch_size=input_tensor.size(0), device=input_tensor.device
        )
        u = [u0,]
        outputs = []
        decay_u = self.tau_u_trainer.get_decay()
        for i in range(input_tensor.size(1)):
            u_tm1 = u[-1]
            current = F.linear(input_tensor[:, i], self.weight, self.bias)
            u_t = decay_u * u_tm1 + (1.0 - decay_u) * current
            outputs.append(u_t)
            u.append(u_t)
        states = torch.stack(u, dim=1).unsqueeze(0)
        outputs = torch.stack(outputs, dim=1)
        return outputs, states

    def apply_parameter_constraints(self):
        self.tau_u_trainer.apply_parameter_constraints()