from typing import Any
import torch
from torch.nn import Module
from module.base_cell import BaseSNN
from module.lif import LIFLayer
from module.alif import ALIFLayer
from module.li import LILayer

class_map: dict[str, BaseSNN] = {
    "LILayer": LILayer,
    "LIFLayer": LIFLayer,
    "ALIFLayer": ALIFLayer,

}


def resolve_model_config(model_config: dict[str, dict]):
    """
    Iterate over layers in the model_config dictonary

    Args:
        model_config (dict[str, dict]): ordered dictonary where the value of each key
        is a dictonary representing the layer class name plus the parameters for this class.
        ex: {
            "layer_1":{"class": "ALIFLayer", "in_features": 10,"out_features": 128, "thr": 0.5}, 
            "layer_2": {"class": "LILayer", "in_features": 128, "out_features": 10
            }
        The layers names are only here to prevent clash of names in the config files
        
    Returns:
        list[Module]: list of instanciated layers
    """
    layers = []
    for i, (k, v) in enumerate(model_config.items()):
        module_class = v["class"]
        # remove the class name from the dictonary of the module parameters
        # as is not part of the parameters and only here to determined the Layer's class
        del v["class"]
        layer_obj = class_map[module_class](**v)
        layers.append(layer_obj)
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
