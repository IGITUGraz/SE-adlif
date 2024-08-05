from torch.nn import Module

import torch
from torch.nn.parameter import Parameter

def get_tau_trainer_class(name: str):
    if name == "interpolation":
        return InterpolationTrainer
    if name == "interpolationCustomGrad":
        return InterpolationCustomGradTrainer
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
        return torch.lerp(self.tau_min, self.tau_max, self.weight)

    def reset_parameters(self):
        torch.nn.init.uniform_(self.weight)
        self.weight.requires_grad = True
        
        
            
class InterpolationCustomGradTrainer(TauTrainer):
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
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__(
            in_features, dt, tau_min, tau_max, device, dtype, **kwargs)
        self.tau_func = TauFunc.apply
        if tau_min == tau_max:
            dw = 0.0
        else:
            dw = 1.0 / (tau_max - tau_min)
        self.dw = torch.tensor(dw, **factory_kwargs)

    def apply_parameter_constraints(self):
        with torch.no_grad():
            self.weight.clamp_(0.0, 1.0)

    def forward(self):
        return self.tau_func(self.weight, self.dw, self.tau_min, self.tau_max, self.dt)
    
    def get_tau(self):

        return torch.lerp(self.tau_min, self.tau_max, self.weight)

    def reset_parameters(self):
        torch.nn.init.uniform_(self.weight)
        self.weight.requires_grad = True

class TauFunc(torch.autograd.Function):
    @staticmethod
    @torch.jit.ignore
    def forward(
        ctx,
        w: torch.Tensor,
        dw: torch.Tensor,
        tau_min: torch.Tensor,
        tau_max: torch.Tensor,
        dt: float
    ) -> torch.Tensor:
        tau = torch.lerp(tau_min, tau_max, w)
        dw_vect = torch.full_like(tau, dw)
        ctx.save_for_backward(dw_vect,)
        return torch.exp(-dt/tau)

    @staticmethod
    @torch.jit.ignore
    def backward(ctx, grad_output):
        (dw,) = ctx.saved_tensors
        grad = dw * grad_output
        return grad, None, None, None, None