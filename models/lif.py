
from typing import Optional
from models.helpers import SLAYER
import torch
import torch.nn.functional as F
from torch.nn import Module
from torch import Tensor
from torch.nn.parameter import Parameter

from module.tau_trainers import TauTrainer, get_tau_trainer_class
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
        in_features: int,
        out_features: int,
        use_recurrent: bool = True,
        thr: float = 1.0,
        alpha: float = 5.0,
        c: float = 0.4,
        tau_u_range: tuple[float, float] = (20, 20),
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
        self.use_recurrent = use_recurrent
        self.thr = thr
        self.alpha = alpha
        self.c = c

        self.weight = Parameter(
            torch.empty((out_features, in_features), **factory_kwargs)
        )
        self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
        if self.use_recurrent:
            self.recurrent = Parameter(
                    torch.empty((out_features, out_features), **factory_kwargs)
                )
        else:
            self.register_buffer("recurrent", None)
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

    def forward(self, input_tensor: Tensor) -> tuple[Tensor, Tensor]:
        
        u0, z0 = self.initial_state(input_tensor.size(0), input_tensor.device)
        u = [u0,]
        z = [z0,]
        outputs = []
        decay_u = self.tau_u_trainer.get_decay()
        for i in range(input_tensor.size(1)):
            u_tm1 = u[-1]
            z_tm1 = z[-1]
            u_tm1 = u_tm1 * (1 - z_tm1.detach())

            soma_current = F.linear(input_tensor[:, i], self.weight, self.bias)
            if self.use_recurrent:
                soma_rec_current = F.linear(z_tm1, self.recurrent, None)
                soma_current += soma_rec_current

            u_t = decay_u * u_tm1 + (1.0 - decay_u) * (soma_current)

            u_thr = u_t - self.thr
            # Forward Gradient Injection trick (credits to Sebastian Otte)
            z_t = torch.heaviside(u_thr, torch.as_tensor(0.0).type(u_thr.dtype)).detach() + (u_thr - u_thr.detach()) * SLAYER(u_thr, self.alpha, self.c).detach()

            outputs.append(z_t)
            u.append(u_t)
            z.append(z_t)
        
        u = torch.stack(u, dim=1)
        z = torch.stack(z, dim=1)
        states = torch.stack([u, z], dim=0)
        outputs = torch.stack(outputs, dim=1)
        return outputs, states


    def apply_parameter_constraints(self):
        self.tau_u_trainer.apply_parameter_constraints()
