from typing import Any
import torch
from torch.nn import Module
from models.alif import SEAdLIF, EFAdLIF
from models.li import LI
from models.lif import LIF
from torch.nn import Dropout


def resolve_model_config(model_config: dict[str, dict]):
    """
    Iterate over layers in the model_config dictonary
        
    Returns:
        list[Module]: list of instanciated layers
    """
    layers = []
    if model_config['model'] in ['SE-adLIF', 'EF-adLIF']:
        main_layer = SEAdLIF if model_config['model'] == 'SE-adLIF' else EFAdLIF
        in_features = model_config['in_features']
        for i in range(model_config['n_layers']):
            layers.append(main_layer(
                in_features=in_features,
                out_features=model_config['n_neurons'],
                alpha=model_config['alpha'],
                c=model_config['c'],
                tau_u_range=model_config['tau_u_range'],
                train_tau_u_method='interpolation',
                tau_w_range=model_config['tau_w_range'],
                train_tau_w_method='interpolation',
                use_recurrent=model_config.get('use_recurrent', True),
                a_range=[0.0, 1.0],
                b_range=[0.0, 2.0],
                q=model_config['q']))
            in_features = model_config['n_neurons']
    #TODO: ADD LSTM wrapper
    else:
        main_layer = LIF
        in_features = model_config['in_features']
        for i in range(model_config['n_layers']):
            layers.append(main_layer(
                in_features=in_features,
                out_features=model_config['n_neurons'],
                alpha=model_config['alpha'],
                c=model_config['c'],
                tau_u_range=model_config['tau_u_range'],
                train_tau_u_method='interpolation',
                use_recurrent=model_config.get('use_recurrent', True),))
            in_features = model_config['n_neurons']
    layers.append(
        LI(
            in_features=model_config['n_neurons'],
            out_features=model_config['out_features'],
            tau_u_range=model_config['tau_out_range'],
            train_tau_u_method=model_config.get('train_tau_out_method', "fixed")
        )
    )
    
    return layers


class CustomSeq(torch.nn.Sequential):
    """
    args: Module list or (str: Module) dictonary
    """

    def __init__(
        self,
        *args: Module
    ):
        super().__init__(*args)
        # get the out_features argument of the last layer
        self.output_size = self._modules.get(
            list(self._modules.keys())[-1]).out_features
        
    def forward(
        self,
        inputs: torch.Tensor,
        record_states: bool = True,
    ):
        states = []
        for module in self:
            inputs, state = module(inputs)
            if record_states:            
                states.append(state) 
        return inputs, states
    
    def apply_parameter_constraints(self):
        for module in self:
            module.apply_parameter_constraints()
