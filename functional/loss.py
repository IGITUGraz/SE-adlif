import torch



def get_per_layer_spike_probs(states_list: list[torch.Tensor], 
                              block_idx):
    """
    Iterate over the recorded states of each layers,
    if this states correspond on to a spiking neurons states.
    Retrieved average spike probability for this layer for regularization purposes.
    
    """
    spike_probs =  [
            get_spike_prob(states[1], block_idx)
            for states in states_list if states.size(0) > 1
        ]
    return spike_probs


def get_spike_prob(z, block_idx):
    """
    Determined the average spike probability for each neuron of a specific layer.
    The spike probability is sum(z^t)/T, where T is the sample duration
    determined as the sum of all valid/non-padded time-steps.
    
    """
    z = z[:, 1:] 
    # create a tensor that have the same shape as z except for the temporal dimension
    # the temporal dimension correspond to result of the scatter operation
    # for the padded timesteps (spike_proba_per_block[:, 0]) and non-padded timesteps
    # (spike_proba_per_block[:, 1]) only the former is of interest to us for the regularization loss
     
    spike_proba_per_block = torch.zeros(
        size=(z.shape[0], 2, z.shape[2]), device=z.device
    )
    # determined all non-padded time-steps
    # assuming that padded time-steps are always summed to block_idx[:, 0]
    padded_timesteps_mask = (block_idx != torch.tensor(0)).long()
    
    spike_proba_per_block.scatter_reduce_(
        1,
        # all padded time-steps are averaged to the 0-index
        # all valid time-steps are averaged to the 1-index
        padded_timesteps_mask.broadcast_to(z.shape),
        z,
        reduce="mean",
        include_self=False,
    )
    # mean over all batches for the non-padded timesteps
    return spike_proba_per_block[:, 1].mean(dim=0) 


def snn_regularization(spike_proba_per_layer: list[torch.Tensor], target: float, reg_type: str):
    """
    Determined the regularization loss 

    Args:
        spike_proba_per_layer (torch.Tensor): spike probability for each neuron per layer
        spike_proba_par_layer[i] is a (nums_neuron, ) tensor representing the spike probability 
        for each neuron of layer i 
        target (float): the spike probability target
        reg_type (str): the regularization type
    
    upper regularization: only regularized neurons where the spike probability is higher than the target
    lower regularization: only regularized neurons where the spike probability is lower than the target 
    both: regularized neuron with respect the squared distance of neuron spike probability and the target

    Raises:
        NotImplementedError: Raise error if reg_type is unknown 

    Returns:
        torch.Tensor: the regularization loss averaged over the total numbers of neurons 
    """
    if reg_type == "lower":
        return torch.mean(torch.concatenate([torch.relu(target - s) ** 2 for s in spike_proba_per_layer]))
    if reg_type == "upper":
        return torch.mean(torch.concatenate([torch.relu(s - target) ** 2 for s in spike_proba_per_layer]))
    if reg_type == "both":
        return torch.mean(torch.concatenate([(s - target) ** 2 for s in spike_proba_per_layer]))
    else:
        raise NotImplementedError(
            f"Regularization type: {reg_type} is not implemented, valid type are [upper, lower, both]"
        )

def calculate_weight_decay_loss(model):
    weight_decay_loss = 0
    for name, param in model.named_parameters():
        if (
            "bias" not in name
            and "tau" not in name
            and "beta" not in name
            and "coeff" not in name
        ):
            weight_decay_loss += torch.mean(param**2)
    return weight_decay_loss
