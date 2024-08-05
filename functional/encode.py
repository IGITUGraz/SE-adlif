from typing import Union, Callable

import torch
import numpy as np

def burst_encode(input_values: torch.Tensor, seq_length: int, delta_min: float,
                 delta_max: float, max_nb_spike: float):
    """
    Encode input value in [0, 1] to a short burst with parametrizable 
    number of spikes and interspike interval (ISI) timing.
    A low value in input result into a low number of spike with high
    ISI and inversly.

    Args:
        input_values (torch.Tensor): real value tensor in [0, 1]
        seq_length (int): time segement in which the burst is considered
        delta_min (float): the minimum delta time between spike
        delta_max (float): the maximum delta time between spike
        max_nb_spike (float): the maximum of spikes in the sequence

    Returns:
        _type_: _description_
    """
    nb_spike = torch.ceil(max_nb_spike*input_values).int()
    isi = torch.ceil(-(delta_max - delta_min) * input_values + delta_max)
    isi = torch.where(nb_spike > 1.0, isi, torch.tensor(delta_max)).int()
    data_shape = isi.reshape(-1).size(0)
    # we have a spike between each isi interval and
    # during the early isi + (isi + 1)*nb_spike ms period
    mask_interval = torch.arange(1, seq_length+1).repeat(data_shape, 1) \
        % (isi.reshape(-1, 1) + 1) == 0
    mask_count = torch.arange(1, seq_length+1).repeat(data_shape, 1) \
        < isi.reshape(-1, 1) + (
            isi.reshape(-1, 1) + 1) * nb_spike.reshape(-1, 1)
    mask = torch.logical_and(mask_interval, mask_count)
    # mask is (data_shape, seq_length)
    # reshape is row major so (seq_length, *inp_shape) will mix everything if 
    # mask is not transpose first
    mask = mask.T
    mask = mask.reshape(shape=(seq_length, *input_values.shape))
    
    return mask.float()


def gaussian_rbf(tensor: torch.Tensor, sigma: float = 1):
    return torch.exp(-tensor / (2 * sigma**2))


def euclidean_distance(x, y):
    return (x - y).pow(2)

def population_encode(
        input_values: torch.Tensor,
        out_features: int,
        min_value: Union[float, torch.Tensor] = None,
        max_value: Union[float, torch.Tensor] = None,
        kernel: Callable[[torch.Tensor], torch.Tensor] = gaussian_rbf,
        distance_function: Callable[
            [torch.Tensor, torch.Tensor], torch.Tensor] = euclidean_distance,
) -> torch.Tensor:
    """Population encoding
    define neurons as centroid of a kernel function.
    Centroid are sampled uniformly into the real line with respect to min/max parameters range of the feature space.
    
    Then for each features present in the input value we determined the spiking probability
    of each neuron for the feature.
    As such each real valued features is encoded by a population of neurons.
    
    Args:
        input_values (torch.Tensor): real valued features tensor of size (*nd, num_features)
        out_features (int): size of the population 
        min_value (Union[float, torch.Tensor], optional): minimum feature value. Defaults to None.
        max_value (Union[float, torch.Tensor], optional): maximum feature value. Defaults to None.
        kernel (Callable[[torch.Tensor], torch.Tensor], optional): kernel function. Defaults to gaussian_rbf.
        distance_function (Callable[ [torch.Tensor, torch.Tensor], torch.Tensor], optional): distance metric. Defaults to euclidean_distance.

    Returns:
        torch.Tensor: a (out_features, num_features)
    """
    # size = (input_values.size(0), out_features) + input_values.size()[1:]
    size = (*input_values.shape, out_features)
    if not min_value:
        min_value = input_values.min()
    if not max_value:
        max_value = input_values.max()
    if out_features == 1:
        center = torch.median(input_values)
        centres = torch.tensor([center], dtype=torch.float).expand(size)
    else:
        centres = torch.linspace(min_value, max_value, out_features).expand(size)
    x = input_values.unsqueeze(-1).expand(size)
    distances = distance_function(x, centres)

    return kernel(distances)


def poisson_encode(input_values: torch.Tensor , seq_length: int,
                   f_max: float = 100, f_min: float = 0.0, 
                   dt: float = 0.001) -> np.ndarray:
    """
    Encodes a tensor of input values, which are assumed to be in the
    range [0,1] into a tensor of one dimension higher of binary values,
    which represent input spikes.
    See for example https://www.cns.nyu.edu/~david/handouts/poisson.pdf.
    Parameters:
        input_values (torch.Tensor): Input data tensor with values
        assumed to be in the interval [0,1].
        seq_length (int): Number of time steps in the resulting spike train.
        f_max (float): Maximal frequency (in Hertz) which will be emitted.
        dt (float): Integration time step
        (should coincide with the integration time step used in the model)
    Returns:
        A tensor with an extra dimension of size
        `seq_length` containing spikes (1) or no spikes (0).
    """
    return (# noqa
        torch.rand(seq_length, *input_values.shape)
        < dt * (f_max * input_values + (1 - input_values)*f_min)
    ).float()
    
