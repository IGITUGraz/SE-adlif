from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Module, Tensor
from torch.nn.parameter import Parameter

from module.base_cell import BaseSNN
from module.tau_trainers import TauTrainer, get_tau_trainer_class
# TODO: remove this comment
# what was removed:
# any tracking hook for plotting
# initialization method: as we use the same things every time
# use bias: as we always use bias
# what was kept:
# tau_mapping: nice encapsulation and 
# we sometime train the tau of the last layer (auto-regression task)

class LILayer(Module):
    __constants__ = ["in_features", "out_features"]
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(
        self,
        in_features: int,
        out_features: int,
        dt: float = 1.0,
        tau: tuple = (20, 20),
        train_tau_method: str = "fixed",
        device=None,
        dtype=None,
        **kwargs,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__(**kwargs)
        self.in_features = in_features
        self.out_features = out_features
        self.dt = dt
        self.tau_range = tau
        self.weight = Parameter(
            torch.empty((out_features, in_features), **factory_kwargs)
        )

        self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
        self.tau_trainer: TauTrainer = get_tau_trainer_class(train_tau_method)(
            out_features,
            self.dt,
            self.tau_range[0],
            self.tau_range[1],
            **factory_kwargs,
        )

        self.reset_parameters()

    def reset_parameters(self):
        self.tau_trainer.reset_parameters()
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
    ) -> torch.Tensor:
        size = (batch_size, self.out_features)
        v = torch.zeros(size=size, device=device, dtype=torch.float, requires_grad=True)
        return v

    def forward(self, input_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # pre-allocate memory for the states and outputs where states is
        # (num_states, batch_size, time, num_neurons)
        # and outputs is (batch_size, time, num_neurons)

        states = torch.empty(
            (1, input_tensor.size(0), input_tensor.size(1) + 1, self.out_features),
            device=input_tensor.device,
        )
        v0 = self.initial_state(
            batch_size=input_tensor.size(0), device=input_tensor.device
        )
        v = [v0,]
        outputs = []
        decay = self.tau_trainer.get_decay()
        for i in range(input_tensor.size(1)):
            v_tm1 = v[-1]
            current = F.linear(input_tensor[:, i], self.weight, self.bias)
            v_t = decay * v_tm1 + (1 - decay) * current
            outputs.append(v_t)
            v.append(v_t)
        states = torch.stack(v, dim=1).unsqueeze(0)
        outputs = torch.stack(outputs, dim=1)
        return outputs, states

    def apply_parameter_constraints(self):
        self.tau_trainer.apply_parameter_constraints()