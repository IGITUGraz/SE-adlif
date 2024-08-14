from typing import Tuple
import torch
from torch.nn import Module
from torch import Tensor
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from models.helpers import SLAYER
from module.tau_trainers import TauTrainer, get_tau_trainer_class
from omegaconf import DictConfig

class EFAdLIF(Module):
    __constants__ = ["in_features", "out_features"]
    in_features: int
    out_features: int
    a: Tensor
    b: Tensor 
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
        self.thr = cfg.get('thr', 1.0)
        self.alpha = cfg.get('alpha', 5.0)
        self.c = cfg.get('c', 0.4)
        self.tau_u_range = cfg.tau_u_range
        self.train_tau_u_method = 'interpolation'
        self.tau_w_range = cfg.tau_w_range
        self.train_tau_w_method = 'interpolation'        
        self.use_recurrent = cfg.get('use_recurrent', True)

        self.a_range = [0.0, 1.0]
        self.b_range = [0.0, 2.0]
        
        self.q = cfg.q
        
        self.tau_u_trainer: TauTrainer = get_tau_trainer_class(self.train_tau_u_method)(
                self.out_features,
                self.dt, 
                self.tau_u_range[0], 
                self.tau_u_range[1],
                **factory_kwargs)
        
        self.tau_w_trainer: TauTrainer = get_tau_trainer_class(self.train_tau_w_method)(
                self.out_features,
                self.dt, 
                self.tau_w_range[0], 
                self.tau_w_range[1],
                **factory_kwargs)
        
        
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

        self.a = Parameter(torch.empty(self.out_features, **factory_kwargs))
        self.b = Parameter(torch.empty(self.out_features, **factory_kwargs))


        self.reset_parameters()
    
    def reset_parameters(self) -> None:
        self.tau_u_trainer.reset_parameters()
        self.tau_w_trainer.reset_parameters()
        
        
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
        
        torch.nn.init.uniform_(self.a, self.a_range[0], self.a_range[1])
        torch.nn.init.uniform_(self.b, self.b_range[0], self.b_range[1])
        
    def initial_state(self, batch_size, device) -> Tensor:
        size = (batch_size, self.out_features)
        u = torch.zeros(
            size=size, 
            device=device, 
            dtype=torch.float, 
            requires_grad=True
        )
        z = torch.zeros(
            size=size, 
            device=device, 
            dtype=torch.float, 
            requires_grad=True
        )
        w = torch.zeros(
            size=size,
            device=device,
            dtype=torch.float,
            requires_grad=True,
        )
        return u, z, w

    def apply_parameter_constraints(self):
        self.tau_u_trainer.apply_parameter_constraints()
        self.tau_w_trainer.apply_parameter_constraints()
        self.a.data = torch.clamp(self.a, min=self.a_range[0], max=self.a_range[1])
        self.b.data = torch.clamp(self.b, min=self.b_range[0], max=self.b_range[1])

    def forward(
        self, input_tensor: Tensor,  states: Tuple[Tensor, Tensor, Tensor]
    ) -> Tuple[Tensor, Tensor]:
        u_tm1, z_tm1, w_tm1 = states
        decay_u = self.tau_u_trainer.get_decay()
        decay_w = self.tau_w_trainer.get_decay()
        soma_current = F.linear(input_tensor, self.weight, self.bias)
        if self.use_recurrent:
            soma_rec_current = F.linear(z_tm1, self.recurrent, None)
            soma_current += soma_rec_current
            
        u_t = decay_u * u_tm1 + (1.0 - decay_u) * (
            soma_current - w_tm1
        )
        
        u_thr = u_t - self.thr
        # Forward Gradient Injection trick (credits to Sebastian Otte)
        z_t = torch.heaviside(u_thr, torch.as_tensor(0.0).type(u_thr.dtype)).detach() + (u_thr - u_thr.detach()) * SLAYER(u_thr, self.alpha, self.c).detach()
        u_t = u_t * (1 - z_t.detach())
        w_t = (
            decay_w * w_tm1
            + (1.0 - decay_w) * (self.a * u_tm1 + self.b * z_tm1) * self.q
        )
        return z_t.clone(), (u_t, z_t, w_t)
    
class SEAdLIF(EFAdLIF):
    def forward(
        self, input_tensor: Tensor, states: Tuple[Tensor, Tensor, Tensor]
    ) -> Tuple[Tensor, Tensor]:
        u_tm1, z_tm1, w_tm1 = states
        decay_u = self.tau_u_trainer.get_decay()
        decay_w = self.tau_w_trainer.get_decay()
        soma_current = F.linear(input_tensor, self.weight, self.bias)
        if self.use_recurrent:
            soma_rec_current = F.linear(z_tm1, self.recurrent, None)
            soma_current += soma_rec_current
            
        u_t = decay_u * u_tm1 + (1.0 - decay_u) * (
            soma_current - w_tm1
        )
        u_thr = u_t - self.thr
        # Forward Gradient Injection trick (credits to Sebastian Otte)
        z_t = torch.heaviside(u_thr, torch.as_tensor(0.0).type(u_thr.dtype)).detach() + (u_thr - u_thr.detach()) * SLAYER(u_thr, self.alpha, self.c).detach()
        
        # Symplectic formulation with early reset

        u_t = u_t * (1 - z_t.detach())
        w_t = (
            decay_w * w_tm1
            + (1.0 - decay_w) * (self.a * u_t + self.b * z_t) * self.q
        )
        return z_t.clone(), (u_t, z_t, w_t)
    