from typing import Tuple
import torch
from torch import Module
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from models.helpers import SLAYER
from module.tau_trainers import TauTrainer, get_tau_trainer_class

class ALIFLayer(Module):
    __constants__ = ["in_features", "out_features"]
    in_features: int
    out_features: int
    param_beta: torch.Tensor
    weight: torch.Tensor

    def __init__(
        self,
        in_features: int,
        out_features: int,
        dt: float = 1.0,
        thr: float = 1.0,
        tau_soma: tuple[float, float] = (20, 20),
        train_soma_tau_method="fixed",  
        tau_adapt: tuple[float, float] = (20, 200),
        train_adapt_tau_method="fixed",

        use_recurrent: bool = True,
        beta: tuple[float, float] = (0.0, 1.0),

        beta2: tuple[float, float] = (0.0, 2.0),

        q: float = 60.0,
        device=None,
        dtype=None,
        **kwargs,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__(**kwargs)
        self.in_features = in_features
        self.out_features = out_features
        self.dt = dt
        self.thr = thr
        self.tau_soma_range = tau_soma
        self.train_soma_tau_method = train_soma_tau_method
        self.tau_adapt_range = tau_adapt
        self.train_adapt_tau_method = train_adapt_tau_method        
        self.use_bias = use_bias
        self.bias_init = bias_init
        self.use_recurrent = use_recurrent

        self.beta_range = beta 
        self.beta2_range = beta2
        self.train_beta = train_beta
        self.train_beta2 = train_beta2
        
        self.beta_init = beta_init
        self.beta2_init = beta2_init
        
        self.initialization_method = initialization_method
        self.initialization_kwargs = initialization_kwargs
        self.surrogate_kwargs = surrogate_kwargs
        self.adapt_coeff = adapt_coeff
        
        self.tau_trainer_soma: TauTrainer = get_tau_trainer_class(train_soma_tau_method)(
                out_features,
                self.dt, self.tau_soma_range[0], self.tau_soma_range[1],
                **factory_kwargs)
        
        self.tau_trainer_adapt: TauTrainer = get_tau_trainer_class(train_adapt_tau_method)(
                out_features,
                self.dt, tau_adapt[0], tau_adapt[1], **factory_kwargs)
        
        
        self.weight = Parameter(
            torch.empty((out_features, in_features), **factory_kwargs)
        )
        if self.use_bias:
            self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_buffer("bias", None)
        if self.use_recurrent:
            self.recurrent = Parameter(
                torch.empty((out_features, out_features), **factory_kwargs)
            )
        else:
            self.register_buffer("recurrent", None)

        self.param_beta = Parameter(torch.empty(out_features, **factory_kwargs))
        self.param_beta2 = Parameter(torch.empty(out_features, **factory_kwargs))


        self.reset_parameters()
    
    def reset_parameters(self) -> None:
        self.tau_trainer_soma.reset_parameters()
        self.tau_trainer_adapt.reset_parameters()
        
        if self.initialization_method == "uniform":
            torch.nn.init.uniform_(
                self.weight,
                -1.0 * torch.sqrt(1 / torch.tensor(self.in_features)),
                torch.sqrt(1 / torch.tensor(self.in_features)),
            )
        elif self.initialization_method == "normal":
            torch.nn.init.normal_(
                self.weight, 0.1, torch.sqrt(1 / torch.tensor(self.in_features))
            )
        elif self.initialization_method == "uniform_half":
            torch.nn.init.uniform_(
                self.weight,
                -1.0 * torch.sqrt(1 / torch.tensor(self.in_features)),
                torch.sqrt(1 / torch.tensor(self.in_features)),
            )
        elif self.initialization_method == "orthogonal":
            torch.nn.init.orthogonal_(
                self.weight, gain=self.initialization_kwargs.get("somatic_gain", 1.0)
            )
        else:
            raise NotImplementedError(
                f"Initialization method {self.initialization_method} not implemented"
            )
        if self.bias is not None:
            torch.nn.init.constant_(self.bias, self.bias_init)
        torch.nn.init.orthogonal_(
            self.recurrent,
            gain=self.initialization_kwargs.get("somatic_rec_gain", 1.0),
        )
        if self.beta_init == "uniform":
            torch.nn.init.uniform_(self.param_beta, self.beta_range[0], self.beta_range[1])
        else:
            torch.nn.init.constant_(self.param_beta, self.beta_init)
        if self.beta2_init == "uniform":
            torch.nn.init.uniform_(self.param_beta2, self.beta2_range[0], self.beta2_range[1])
        else:
            torch.nn.init.constant_(self.param_beta2, self.beta2_init)

        self.param_beta.requires_grad_(self.train_beta)
        self.param_beta2.requires_grad_(self.train_beta2)

    def initial_state(self, batch_size, device) -> torch.Tensor:
        size = (batch_size, self.out_features)
        vs=torch.zeros(
            size=size, 
            device=device, 
            dtype=torch.float, 
            requires_grad=True
        )
        z=torch.zeros(
            size=size, 
            device=device, 
            dtype=torch.float, 
            requires_grad=True
        )
        b=torch.zeros(
            size=size,
            device=device,
            dtype=torch.float,
            requires_grad=True,
        )
        return vs, z, b

    def apply_parameter_constraints(self):
        self.tau_trainer_soma.apply_parameter_constraints()
        self.tau_trainer_adapt.apply_parameter_constraints()
        self.param_beta.data = torch.clamp(self.param_beta, min=self.beta_range[0], max=self.beta_range[1])
        self.param_beta2.data = torch.clamp(self.param_beta2, min=self.beta2_range[0], max=self.beta2_range[1])

    def forward(
        self, input_tensor: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        decay_v = self.tau_trainer_soma.get_decay()
        decay_b = self.tau_trainer_adapt.get_decay()
        v0, z0, b0 = self.initial_state(input_tensor.size(0), input_tensor.device)
        v = [v0,]
        z = [z0,]
        b = [b0,]
        outputs = []
        beta = self.param_beta
        beta2 = self.param_beta2

        for i in range(input_tensor.size(1)):
            v_tm1 = v[-1]
            z_tm1 = z[-1]
            b_tm1 = b[-1]
            v_tm1 = self.reset_func(v_tm1, z_tm1)
            soma_current = F.linear(input_tensor[:, i], self.weight, self.bias)
            if self.use_recurrent:
                soma_rec_current = F.linear(z_tm1, self.recurrent, None)
                soma_current += soma_rec_current
                
            v_t = decay_v * v_tm1 + (1.0 - decay_v) * (
                soma_current - b_tm1
            )
            
            v_thr = v_t - self.thr
            # Forward Gradient Injection trick (credits to Sebastian Otte)
            z_t = torch.heaviside(v_thr, 1.0).detach() + (v_thr - v_thr.detach()) * SLAYER(v_thr, self.alpha, self.c).detach()
            
            b_t = (
                decay_b * b_tm1
                + (1 - decay_b) * (beta * v_t + beta2 * z_t) * self.adapt_coeff
            )
            
            outputs.append(z_t)
            v.append(v_t)
            z.append(z_t)
            b.append(b_t)
        v = torch.stack(v, dim=1)
        z = torch.stack(z, dim=1)
        b = torch.stack(b, dim=1)
        states = torch.stack([v, z, b], dim=0)
        outputs = torch.stack(outputs, dim=1)
        return outputs, states