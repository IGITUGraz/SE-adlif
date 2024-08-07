from typing import Optional
import torch
from torch.nn import LSTMCell
class LSTMCellWrapper(LSTMCell):
    def __init__(self,cfg ):
        super().__init__(input_size=cfg.input_size, hidden_size=cfg.n_neurons)
        h_size = self.hidden_size
        forget_start = h_size//4
        forget_end = h_size//2
        forget_bias_init = cfg.get('forget_bias_init', None)
        if forget_bias_init is not None:
            self.bias_ih.data[forget_start:forget_end].fill_(forget_bias_init)
            
    def forward(self, inputs, states):
        states = super().forward(inputs, (states[0], states[1]))
        return states[0], (states[0], states[1])
    def initial_state(self, batch_size: int, device: Optional[torch.device] = None):
        size = (batch_size, self.hidden_size)
        zeros = torch.zeros(
            size=size, dtype=torch.float, device=device
        )
        return (zeros, zeros)
    def apply_parameter_constraints(self):
        pass