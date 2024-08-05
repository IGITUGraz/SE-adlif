import torch
from torch.nn import Module
from typing import NamedTuple


class BaseSNN(Module):
    def __init__(self,) -> None:
        super().__init__()
            
    def threshold_func(self, v, surrogate_kwargs: dict):
        alpha = surrogate_kwargs.get('alpha', 4.9)
        scale = surrogate_kwargs.get('scale', 0.4)
        return Hyperbolic.apply(v, alpha, scale)
    
    def reset_func(self, v: torch.tensor, z: torch.Tensor):
        return v*(1 - z.detach())
        
    def initial_state(self, batch_size:int, device: torch.device
                      ) -> NamedTuple:
        """Initialized states for the cell

        Args:
            batch_size (int): the batch size
            device (torch.device): memory device location

        Raises:
            NotImplementedError: This function 
            should be implemented in concrete sub-class

        Returns:
            NamedTuple: state
        """
        raise NotImplementedError("This function should not be call from baseSNNCell instance")
    def apply_parameter_constraints(self):
        pass

class Hyperbolic(torch.autograd.Function): 
    @staticmethod
    @torch.jit.ignore
    def forward(ctx, x: torch.Tensor, alpha: float, scale: float) -> torch.Tensor:
        ctx.save_for_backward(x)
        ctx.alpha = alpha
        ctx.scale = scale
        return torch.heaviside(x, torch.as_tensor(0.0).type(x.dtype))

    @staticmethod
    @torch.jit.ignore
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        alpha = ctx.alpha
        scale = ctx.scale
        grad_input = grad_output.clone()
        grad = grad_input * scale * alpha / (2 * torch.exp(x.abs() * alpha))
        return grad, None, None
    