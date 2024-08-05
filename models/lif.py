
from typing import Optional
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn.parameter import Parameter

from module.base_cell import BaseSNN
from module.tau_trainers import TauTrainer, get_tau_trainer_class
from utils.utils import save_distributions_to_aim, save_fig_to_aim, get_event_indices


class LIFLayer(BaseSNN):
    __constants__ = ["in_features", "out_features"]
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(
        self,
        in_features: int,
        out_features: int,
        thr: float = 1.0,
        tau_soma: tuple[float, float] = (20, 20),
        train_soma_tau_method: str = "fixed",
        use_bias: bool = True,
        bias_init: float = 0.0,
        use_recurrent: bool = True,
        alpha: float = 5.0,
        c: float = 0.4,
        device=None,
        dtype=None,
        **kwargs,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__(**kwargs)
        self.in_features = in_features
        self.out_features = out_features
        self.dt = 1
        self.tau_soma_range = tau_soma
        self.use_bias = use_bias
        self.use_recurrent = use_recurrent
        self.thr = thr
        self.alpha = alpha
        self.c = c
        self.bias_init = bias_init
        
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
        self.reset_parameters()

    def reset_parameters(self):
        self.tau_trainer.reset_parameters()
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
        # The following is not called, when there is no recurrent connections
        torch.nn.init.orthogonal_(
            self.recurrent,
            gain=self.initialization_kwargs.get("somatic_rec_gain", 1.0),
        )

    def initial_state(
        self, batch_size: int, device: Optional[torch.device] = None
    ) -> torch.Tensor:
        size = (batch_size, self.out_features)
        vs = torch.zeros(
            size=size, device=device, dtype=torch.float, requires_grad=True
        )
        z = torch.zeros(size=size, device=device, dtype=torch.float, requires_grad=True)

        return vs, z

    def forward(self, input_tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # pre-allocate memory for the states and outputs where states is
        # (num_states, batch_size, time, num_neurons)
        # and outputs is (batch_size, time num_neurons)
        
        v0, z0 = self.initial_state(input_tensor.size(0), input_tensor.device)
        v = [v0,]
        z = [z0,]
        outputs = []
        decay_vs = self.tau_trainer.get_decay()
        for i in range(input_tensor.size(1)):
            v_tm1 = v[-1]
            z_tm1 = z[-1]
            v_tm1 = self.reset_func(v_tm1, z_tm1)

            soma_current = F.linear(input_tensor[:, i], self.weight, self.bias)
            if self.use_recurrent:
                soma_rec_current = F.linear(z_tm1, self.recurrent, None)
                soma_current += soma_rec_current

            v_t = decay_vs * v_tm1 + (1.0 - decay_vs) * (soma_current)
            v_scaled = (v_t - self.thr) / self.thr

            z_t = self.threshold_func(v_scaled, self.surrogate_kwargs)

            outputs.append(z_t)
            v.append(v_t)
            z.append(z_t)
        
        v = torch.stack(v, dim=1)
        z = torch.stack(z, dim=1)
        states = torch.stack([v, z], dim=0)
        outputs = torch.stack(outputs, dim=1)
        return outputs, states

    @torch.jit.export
    def extra_repr(self) -> str:
        return "in_features={}, out_features={}, bias={}, rec={}".format(
            self.in_features,
            self.out_features,
            self.bias is not None,
            self.recurrent is not None,
        )

    @staticmethod
    def plot_states(layer_idx, inputs, states):
        figure, axes = plt.subplots(nrows=3, ncols=1, sharex="all", figsize=(8, 11))
        inputs = inputs.cpu().detach().numpy()
        states = states.cpu().detach().numpy()
        axes[0].eventplot(
            get_event_indices(inputs.T), color="black", orientation="horizontal"
        )
        axes[0].set_ylabel("input")
        axes[1].plot(states[0])
        axes[1].set_ylabel("v_t")
        axes[2].eventplot(
            get_event_indices(states[1].T), color="black", orientation="horizontal"
        )
        axes[2].set_ylabel("z_t/output")
        nb_spikes_str = str(states[1].sum())
        figure.suptitle(f"Layer {layer_idx}\n Nb spikes: {nb_spikes_str},")
        plt.close(figure)
        return figure

    @torch.jit.ignore
    def layer_stats(
        self,
        layer_idx: int,
        logger,
        epoch_step: int,
        spike_probabilities: torch.Tensor,
        inputs: torch.Tensor,
        states: torch.Tensor,
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
        """

        save_fig_to_aim(
            logger=logger,
            name=f"{layer_idx}_Activity",
            figure=LIFLayer.plot_states(layer_idx, inputs, states),
            epoch_step=epoch_step,
        )

        distributions = [
            ("soma_tau", self.tau_trainer.get_tau().cpu().detach().numpy()),
            ("soma_weights", self.weight.cpu().detach().numpy()),
            ("spike_prob", spike_probabilities.cpu().detach().numpy()),
            ("bias", self.bias.cpu().detach().numpy()),
        ]

        if self.use_recurrent:
            distributions.append(
                ("recurrent_weights", self.recurrent.cpu().detach().numpy())
            )
        save_distributions_to_aim(
            logger=logger,
            distributions=distributions,
            name=f"{layer_idx}",
            epoch_step=epoch_step,
        )

    def apply_parameter_constraints(self):
        self.tau_trainer.apply_parameter_constraints()
