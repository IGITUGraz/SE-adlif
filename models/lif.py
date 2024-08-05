
from typing import Optional
import matplotlib.pyplot as plt
from models.helpers import SLAYER
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn.parameter import Parameter

from module.base_cell import BaseSNN
from module.tau_trainers import TauTrainer, get_tau_trainer_class
from utils.utils import save_distributions_to_aim, save_fig_to_aim, get_event_indices


class LIFLayer(BaseSNN):
    __constants__ = ["in_features", "out_features"]
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(
        self,
        in_features: int,
        out_features: int,
        thr: float = 1.0,
        tau_soma: tuple[float, float] = (20, 20),
        train_soma_tau_method: str = "fixed",
        use_bias: bool = True,
        bias_init: float = 0.0,
        use_recurrent: bool = True,
        alpha: float = 5.0,
        c: float = 0.4,
        device=None,
        dtype=None,
        **kwargs,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__(**kwargs)
        self.in_features = in_features
        self.out_features = out_features
        self.dt = 1
        self.tau_soma_range = tau_soma
        self.use_bias = use_bias
        self.thr = thr
        self.alpha = alpha
        self.c = c

        self.weight = Parameter(
            torch.empty((out_features, in_features), **factory_kwargs)
        )
        self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
        self.recurrent = Parameter(
                torch.empty((out_features, out_features), **factory_kwargs)
            )
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.uniform_(
            self.weight,
            -1.0 * torch.sqrt(1 / torch.tensor(self.in_features)),
            torch.sqrt(1 / torch.tensor(self.in_features)),
        )
        torch.nn.init.constant_(self.bias, self.bias_init)
        torch.nn.init.orthogonal_(
            self.recurrent,
            gain=1.0,
        )

    def initial_state(
        self, batch_size: int, device: Optional[torch.device] = None
    ) -> torch.Tensor:
        size = (batch_size, self.out_features)
        vs = torch.zeros(
            size=size, device=device, dtype=torch.float, requires_grad=True
        )
        z = torch.zeros(size=size, device=device, dtype=torch.float, requires_grad=True)
        return vs, z

    def forward(self, input_tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # pre-allocate memory for the states and outputs where states is
        # (num_states, batch_size, time, num_neurons)
        # and outputs is (batch_size, time num_neurons)
        
        v0, z0 = self.initial_state(input_tensor.size(0), input_tensor.device)
        v = [v0,]
        z = [z0,]
        outputs = []
        decay_vs = self.tau_trainer.get_decay()
        for i in range(input_tensor.size(1)):
            v_tm1 = v[-1]
            z_tm1 = z[-1]
            v_tm1 = v_tm1 * (1 - z_tm1)

            soma_current = F.linear(input_tensor[:, i], self.weight, self.bias)
            soma_rec_current = F.linear(z_tm1, self.recurrent, None)
            soma_current += soma_rec_current

            v_t = decay_vs * v_tm1 + (1.0 - decay_vs) * (soma_current)

            v_thr = v_t - self.thr
            # Forward Gradient Injection trick (credits to Sebastian Otte)
            z_t = torch.heaviside(v_thr, 1.0).detach() + (v_thr - v_thr.detach()) * SLAYER(v_thr, self.alpha, self.c).detach()

            outputs.append(z_t)
            v.append(v_t)
            z.append(z_t)
        
        v = torch.stack(v, dim=1)
        z = torch.stack(z, dim=1)
        states = torch.stack([v, z], dim=0)
        outputs = torch.stack(outputs, dim=1)
        return outputs, states


    def apply_parameter_constraints(self):
        self.
