from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.parameter import Parameter

from module.base_cell import BaseSNN
from module.tau_trainers import TauTrainer, get_tau_trainer_class
from utils.utils import get_event_indices, save_distributions_to_aim, save_fig_to_aim


class LILayer(BaseSNN):
    __constants__ = ["in_features", "out_features"]
    in_features: int
    out_features: int
    weight: Tensor
    param_decay: Tensor

    def __init__(
        self,
        in_features: int,
        out_features: int,
        dt: float = 1.0,
        tau: tuple = (20, 20),
        use_bias: bool = True,
        train_tau_method: str = "fixed",
        initialization_method: str = "normal",
        device=None,
        dtype=None,
        **kwargs,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__(**kwargs)
        self.in_features = in_features
        self.out_features = out_features
        self.dt = dt
        self.tau_range = tau
        self.use_bias = use_bias
        self.initialization_method = initialization_method
        self.weight = Parameter(
            torch.empty((out_features, in_features), **factory_kwargs)
        )
        if self.use_bias:
            self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_buffer("bias", None)
        self.tau_trainer: TauTrainer = get_tau_trainer_class(train_tau_method)(
            out_features,
            self.dt,
            self.tau_range[0],
            self.tau_range[1],
            **factory_kwargs,
        )

        self.reset_parameters()

    def reset_parameters(self):
        self.tau_trainer.reset_parameters()
        if self.initialization_method == "uniform":
            nn.init.uniform_(
                self.weight,
                -1 * torch.sqrt(1 / torch.tensor(self.in_features)),
                torch.sqrt(1 / torch.tensor(self.in_features)),
            )
        elif self.initialization_method == "normal":
            nn.init.normal_(
                self.weight, 0.0, torch.sqrt(1 / torch.tensor(self.in_features))
            )
        elif self.initialization_method == "orthogonal":
            nn.init.orthogonal_(self.weight)
        elif self.initialization_method == "xavier_normal":
            nn.init.xavier_normal_(self.weight)
        else:
            raise ValueError(
                f"Unknown initialization method {self.initialization_method}"
            )

        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)

    @torch.jit.ignore
    def extra_repr(self) -> str:
        return "in_features={}, out_features={}, bias={}".format(
            self.in_features, self.out_features, self.bias is not None
        )

    def initial_state(
        self, batch_size: int, device: Optional[torch.device] = None
    ) -> torch.Tensor:
        size = (batch_size, self.out_features)
        v = torch.zeros(size=size, device=device, dtype=torch.float, requires_grad=True)
        return v

    def forward(self, input_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # pre-allocate memory for the states and outputs where states is
        # (num_states, batch_size, time, num_neurons)
        # and outputs is (batch_size, time, num_neurons)

        states = torch.empty(
            (1, input_tensor.size(0), input_tensor.size(1) + 1, self.out_features),
            device=input_tensor.device,
        )
        v0 = self.initial_state(
            batch_size=input_tensor.size(0), device=input_tensor.device
        )
        v = [v0,]
        outputs = []
        decay = self.tau_trainer.get_decay()
        for i in range(input_tensor.size(1)):
            v_tm1 = v[-1]
            current = F.linear(input_tensor[:, i], self.weight, self.bias)
            v_t = decay * v_tm1 + (1 - decay) * current
            outputs.append(v_t)
            v.append(v_t)
        states = torch.stack(v, dim=1).unsqueeze(0)
        outputs = torch.stack(outputs, dim=1)
        return outputs, states

    def apply_parameter_constraints(self):
        self.tau_trainer.apply_parameter_constraints()

    @staticmethod
    def plot_states(layer_idx, inputs, states, targets, block_idx, output_size):
        # the li layer is always assumed to be the output layer of a classification
        # problem the argmax of the state is thus compared this respect to the target
        # if the problem would be a regression this code should be changed

        figure, axes = plt.subplots(nrows=3, ncols=1, sharex="all", figsize=(8, 11))
        inputs = inputs.cpu().detach().numpy()
        # remove the first states as it's the initialization states
        states = states[:, 1:].cpu().detach().numpy()
        targets = targets.cpu().detach().numpy()
        block_idx = block_idx.cpu().detach().numpy()
        targets_in_time = targets[block_idx]

        axes[0].eventplot(
            get_event_indices(inputs.T), color="black", orientation="horizontal"
        )
        axes[0].set_ylabel("Input")
        axes[1].plot(states[0])
        axes[1].set_ylabel("v_t/output")
        pred = np.argmax(states[0], -1)
        axes[2].plot(pred, color="blue", label="Prediction")
        axes[2].plot(targets_in_time, color="red", label="Target")
        axes[2].legend()
        axes[2].set_ylabel("Class")
        figure.suptitle(f"Layer {layer_idx}\n")
        plt.tight_layout()
        plt.close(figure)
        return figure

    def layer_stats(
        self,
        layer_idx: int,
        logger,
        epoch_step: int,
        inputs: torch.Tensor,
        states: torch.Tensor,
        targets: torch.Tensor,
        block_idx: torch.Tensor,
        output_size: int,
        **kwargs,
    ):
        """Generate statistisc from the layer weights and a plot of the layer dynamics for a random task example
        Args:
            layer_idx (int): index for the layer in the hierarchy
            logger (_type_): aim logger reference
            epoch_step (int): epoch
            spike_probability (torch.Tensor): spike probability for each neurons
            inputs (torch.Tensor): random example
            states (torch.Tensor): states associated to the computation of the random example
            targets (torch.Tensor): target associated to the random example
            block_idx (torch.Tensor): block indices associated to the random example
        """
        save_fig_to_aim(
            logger=logger,
            name=f"{layer_idx}_Activity",
            figure=LILayer.plot_states(
                layer_idx, inputs, states, targets, block_idx, output_size,
            ),
            epoch_step=epoch_step,
        )

        distributions = [
            ("tau", self.tau_trainer.get_tau().cpu().detach().numpy()),
            ("weights", self.weight.cpu().detach().numpy()),
            ("bias", self.bias.cpu().detach().numpy()),
        ]

        save_distributions_to_aim(
            logger=logger,
            distributions=distributions,
            name=f"{layer_idx}",
            epoch_step=epoch_step,
        )
