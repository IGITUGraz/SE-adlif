import torch
from torch.nn.utils.rnn import pad_sequence



class PadTensors:
    """This is a custom collate function for a pytorch dataloader to load multiple
    event recordings at once. It's intended to be used in combination with sparse tensors.
    All tensor sizes are extended to the largest one in the batch, i.e. the longest recording.
    
    use rnn.pad_sequence that call a C complied function
    """

    def __init__(self, batch_first: bool = True):
        self.batch_first = batch_first
        self.batch_dim = 0 if batch_first else 1
        self.temp_dim = 1 if batch_first else 0


    @staticmethod
    def _call(batch, batch_first: bool):        
        inputs, target_list, block_idx = list(zip(*batch))  # type: ignore
        
        # # add a dummy block if their is one "target" block per sample
        # if len(target_list[0].shape) == 0:
        #     target_list = torch.tensor(target_list).unsqueeze(1)

        target = pad_sequence(
        target_list, batch_first=batch_first, padding_value=-1) 

        # Padding block MUST have target of -1 !!
        target = torch.concatenate((torch.full((target.shape[0], 1), fill_value=-1), target), dim=1)

        inputs = pad_sequence(inputs, batch_first=batch_first, padding_value=0)

        block_idx = pad_sequence(
            block_idx, batch_first=batch_first, padding_value=0
            ).long()
        return inputs, target, block_idx

    def __call__(self, batch):
        return self._call(batch=batch, batch_first=self.batch_first)