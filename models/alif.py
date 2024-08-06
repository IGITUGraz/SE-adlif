from typing import Tuple
import torch
from torch.nn import Module
from torch import Tensor
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from models.helpers import SLAYER
from module.tau_trainers import TauTrainer, get_tau_trainer_class

class EFAdLIF(Module):
    __constants__ = ["in_features", "out_features"]
    in_features: int
    out_features: int
    a: Tensor
    b: Tensor 
    weight: Tensor

    def __init__(
        self,
        in_features: int,
        out_features: int,
        thr: float = 1.0,
        alpha: float = 5.0,
        c: float = 0.4,
        tau_u_range: tuple[float, float] = (20, 20),
        train_tau_u_method: str = "fixed",  
        tau_w_range: tuple[float, float] = (20, 200),
        train_tau_w_method: str = "fixed",
        use_recurrent: bool = True,
        a_range: tuple[float, float] = (0.0, 1.0),
        b_range: tuple[float, float] = (0.0, 2.0),
        q: float = 60.0,
        device=None,
        dtype=None,
        **kwargs,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__(**kwargs)
        self.in_features = in_features
        self.out_features = out_features
        self.dt = 1.0
        self.thr = thr
        self.alpha = alpha
        self.c = c
        self.tau_u_range = tau_u_range
        self.train_tau_u_method = train_tau_u_method
        self.tau_w_range = tau_w_range
        self.train_tau_w_method = train_tau_w_method        
        self.use_recurrent = use_recurrent

        self.a_range = a_range 
        self.b_range = b_range
        
        self.q = q
        
        self.tau_u_trainer: TauTrainer = get_tau_trainer_class(train_tau_u_method)(
                out_features,
                self.dt, 
                self.tau_u_range[0], 
                self.tau_u_range[1],
                **factory_kwargs)
        
        self.tau_w_trainer: TauTrainer = get_tau_trainer_class(train_tau_w_method)(
                out_features,
                self.dt, 
                tau_w_range[0], 
                tau_w_range[1],
                **factory_kwargs)
        
        
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

        self.a = Parameter(torch.empty(out_features, **factory_kwargs))
        self.b = Parameter(torch.empty(out_features, **factory_kwargs))


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
        self, input_tensor: Tensor
    ) -> Tuple[Tensor, Tensor]:
        decay_u = self.tau_u_trainer.get_decay()
        decay_w = self.tau_w_trainer.get_decay()
        u0, z0, w0 = self.initial_state(input_tensor.size(0), input_tensor.device)
        u_tm1 = u0
        z_tm1 = z0
        w_tm1 = w0
        outputs = []
        a = self.a
        b = self.b

        for i in range(input_tensor.size(1)):
            u_tm1 = u_tm1 * (1 - z_tm1.detach())
            soma_current = F.linear(input_tensor[:, i], self.weight, self.bias)
            if self.use_recurrent:
                soma_rec_current = F.linear(z_tm1, self.recurrent, None)
                soma_current += soma_rec_current
                
            u_t = decay_u * u_tm1 + (1.0 - decay_u) * (
                soma_current - w_tm1
            )
            
            u_thr = u_t - self.thr
            # Forward Gradient Injection trick (credits to Sebastian Otte)
            z_t = torch.heaviside(u_thr, torch.as_tensor(0.0).type(u_thr.dtype)).detach() + (u_thr - u_thr.detach()) * SLAYER(u_thr, self.alpha, self.c).detach()
            
            w_t = (
                decay_w * w_tm1
                + (1.0 - decay_w) * (a * u_tm1 + b * z_tm1) * self.q
            )
            
            outputs.append(z_t)
            u_tm1 = u_t
            z_tm1 = z_t
            w_tm1 = w_t
        outputs = torch.stack(outputs, dim=1)
        return outputs
class SEAdLIF(EFAdLIF):
    def forward(
        self, input_tensor: Tensor
    ) -> Tuple[Tensor, Tensor]:
        decay_u = self.tau_u_trainer.get_decay()
        decay_w = self.tau_w_trainer.get_decay()
        u0, z0, w0 = self.initial_state(input_tensor.size(0), input_tensor.device)
        u_tm1 = u0
        z_tm1 = z0
        w_tm1 = w0
        outputs = []
        a = self.a
        b = self.b

        for i in range(input_tensor.size(1)):
            soma_current = F.linear(input_tensor[:, i], self.weight, self.bias)
            if self.use_recurrent:
                soma_rec_current = F.linear(z_tm1, self.recurrent, None)
                soma_current += soma_rec_current
                
            u_t = decay_u * u_tm1 + (1.0 - decay_u) * (
                soma_current - w_tm1
            )
            u_thr = u_t - self.thr
            # Forward Gradient Injection trick (credits to Sebastian Otte, arXiv:2406.00177)
            z_t = torch.heaviside(u_thr, torch.as_tensor(0.0).type(u_thr.dtype)).detach() + (u_thr - u_thr.detach()) * SLAYER(u_thr, self.alpha, self.c).detach()
            # Symplectic formulation with early reset
            u_t = u_t * (1.0 - z_t.detach())
            w_t = (
                decay_w * w_tm1
                + (1.0 - decay_w) * (a * u_t + b * z_t) * self.q
            )
            
            outputs.append(z_t)
            u_tm1 = u_t
            z_tm1 = z_t
            w_tm1 = w_t
        outputs = torch.stack(outputs, dim=1)
        return outputs
    