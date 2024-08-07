from torch.nn import Module

import torch
from torch.nn.parameter import Parameter

def get_tau_trainer_class(name: str):
    if name == "interpolation":
        return InterpolationTrainer
    elif name == "fixed":
        return FixedTau
    else:
        raise ValueError("Invalid tau trainer name: " + name)


class TauTrainer(Module):
    __constants__ = ["in_features"]
    weight: torch.Tensor
    def __init__(
        self,
        in_features: int,        
        dt: float,
        tau_min: float,
        tau_max: float,
        device=None,
        dtype=None,
        **kwargs
        ) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(TauTrainer, self).__init__(**kwargs)
        self.dt = dt        
        self.weight = Parameter(torch.empty(in_features, **factory_kwargs))
        self.register_buffer("tau_max", torch.tensor(tau_max, **factory_kwargs))
        self.register_buffer("tau_min", torch.tensor(tau_min, **factory_kwargs))
        
        
        
    def reset_parameters(self) -> None:
        raise NotImplementedError("This function should not be call from the base class.")
    
    def apply_parameter_constraints(self) -> None:
        raise NotImplementedError("This function should not be call from the base class.")
    
    def foward(self) -> torch.Tensor:
        raise  NotImplementedError("This function should not be call from the base class.")
    
    def get_tau(self) -> torch.Tensor:
        raise  NotImplementedError("This function should not be call from the base class.")
    
    def get_decay(self):
        return self.forward()

class FixedTau(TauTrainer):
    def __init__(
        self,
        in_features: int,
        dt: float,
        tau_min: float,
        tau_max: float,
        device=None,
        dtype=None,
        **kwargs,
    ) -> None:
        super(FixedTau, self).__init__(
            in_features, dt, tau_min, tau_max, device, dtype, **kwargs)
        
    def apply_parameter_constraints(self):
        pass

    def forward(self):
        return torch.exp(-self.dt / self.get_tau())

    def get_tau(self):
        return self.weight

    def reset_parameters(self):
        torch.nn.init.uniform_(self.weight, a=self.tau_min, b=self.tau_max)
        self.weight.requires_grad = False
        


class InterpolationTrainer(TauTrainer):
    def __init__(
        self,
        in_features: int,
        dt: float,
        tau_min: float,
        tau_max: float,
        device=None,
        dtype=None,
        **kwargs,
    ) -> None:
        super().__init__(in_features, dt, tau_min, tau_max, device, dtype, **kwargs)
    def apply_parameter_constraints(self):
        with torch.no_grad():
            self.weight.clamp_(0.0, 1.0)

    def forward(self):
        return torch.exp(-self.dt / self.get_tau())

    def get_tau(self):
        return  self.weight * self.tau_max + (1.0 - self.weight) * self.tau_min

    def reset_parameters(self):
        torch.nn.init.uniform_(self.weight)
        self.weight.requires_grad = True
