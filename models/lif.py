
from typing import Optional
from models.helpers import SLAYER
import torch
import torch.nn.functional as F
from torch.nn import Module
from torch import Tensor
from torch.nn.parameter import Parameter

from module.tau_trainers import TauTrainer, get_tau_trainer_class
from omegaconf import DictConfig
# TODO: remove this comment
# what was removed:
# any tracking hook for ploting
# initialization method: as we use the same things every time
# use bias: as we always use bias
# what was kept:
# tau_mapping: nice encapsulation 
# use_recurrent: some experiments compare removing the recurrent connection 
# keeping this possibility is necessary to reproduce results
# what was modify:
# hardcoded slayer as "gradient injection"


class LIF(Module):
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
        self.in_features = cfg.input_size
        self.out_features = cfg.n_neurons
        self.dt = 1.0
        self.tau_u_range = cfg.get('tau_u_range', [20, 20])
        self.train_tau_u_method = cfg.get('train_tau_u_method', 'interpolation')
        
        self.use_recurrent = cfg.get('use_recurrent', True)
        self.thr = cfg.get('thr', 1.0)
        
        self.alpha = cfg.get('alpha', 5.0)
        self.c = cfg.get('c', 0.4)

        self.weight = Parameter(
            torch.empty((self.out_features, self.in_features), **factory_kwargs)
        )
        self.bias = Parameter(torch.empty(self.out_features, **factory_kwargs))
        if self.use_recurrent:
            self.recurrent = Parameter(
                    torch.empty((self.out_features, self.out_features), **factory_kwargs)
                )
        else:
            self.register_buffer("recurrent", None)
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
        torch.nn.init.uniform_(
            self.weight,
            -1.0 * torch.sqrt(1 / torch.tensor(self.in_features)),
            torch.sqrt(1 / torch.tensor(self.in_features)),
        )
        torch.nn.init.zeros_(self.bias)
        if self.use_recurrent:
            torch.nn.init.orthogonal_(
                self.recurrent,
                gain=1.0,
            )

    def initial_state(
        self, batch_size: int, device: Optional[torch.device] = None
    ) -> Tensor:
        size = (batch_size, self.out_features)
        u = torch.zeros(
            size=size, device=device, dtype=torch.float, requires_grad=True
        )
        z = torch.zeros(size=size, device=device, dtype=torch.float, requires_grad=True)
        return u, z

    def forward(self, input_tensor: Tensor, states: tuple[Tensor, Tensor]) -> tuple[Tensor, Tensor]:
        u_tm1, z_tm1 = states
        decay_u = self.tau_u_trainer.get_decay()
        soma_current = F.linear(input_tensor, self.weight, self.bias)
        if self.use_recurrent:
            soma_rec_current = F.linear(z_tm1, self.recurrent, None)
            soma_current += soma_rec_current

        u_t = decay_u * u_tm1 + (1.0 - decay_u) * (soma_current)
        u_thr = u_t - self.thr
        # Forward Gradient Injection trick (credits to Sebastian Otte)
        z_t = torch.heaviside(u_thr, torch.as_tensor(0.0).type(u_thr.dtype)).detach() + (u_thr - u_thr.detach()) * SLAYER(u_thr, self.alpha, self.c).detach()
        u_t = u_t * (1 - z_t.detach())
        return z_t, (u_t, z_t)


    def apply_parameter_constraints(self):
        self.tau_u_trainer.apply_parameter_constraints()
