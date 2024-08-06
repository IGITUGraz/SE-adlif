# SE-adlif
Shield: [![CC BY-SA 4.0][cc-by-sa-shield]][cc-by-sa]

# Baronig, Ferrand, Sabathiel & Legenstein 2024: Advancing Spatio-Temporal Processing in Spiking Neural Networks through Adaptation

## Getting Started

Install dependencies 

`pip install -r requirements.yml`

## Reproducing results

Start the corresponding experiment with

`python run.py experiment <experiment name> ++logdir path/to/my/logdir ++datadir path/to/my/datadir`

where experiment name corresponds to one of the experiments in the config/experiment folder.

For configuration, we use [Hydra](https://hydra.cc/). To override any parameter, use the ++ syntax. Example to override number of training epochs:

`python run.py experiment SHD_SE_adLIF_small ++logdir path/to/my/logdir ++datadir path/to/my/datadir ++n_epochs=10`

## Important infos

### Block layout

In some tasks (e.g. SHD and SSC) we have to deal with different-length sequences within the same minibatch. We handle this case by a custom masking procedure, using a block index array (block_idx) for each data sample, which acts similar to a mask. We append zeros to samples of shorter sequence length to ensure uniform sequence length within a batch, but mask the padded timesteps with zeros in the block_idx array. The value in the block_idx array then gives the corresponding target class in the target vector. Example

```
data vector: |1011010100101001010000000000000|
             |-----data---------|--padding---| 
             ---> time

block_idx:   |1111111111111111111100000000000|
target: [-1, 3]
```

The data vector contains a block of data, concatenated with zeros to match the length of the longest sequence item in the minibatch.
The block_idx contains ones at the time steps where data is and zeros at padded time steps.
The target is a vector such that target[block_idx] gives the target of the block. In this example, block 0 has target -1 which is ignored, and block 1 (which is the valid data) has target of class 3.

We use this structure to also support per-timestep labels as for example in the ECG task. Example:
```
data vector: |1 0 1 1 0 0 1 0 0 0 0 0 0|
             |-----data---|--padding---| 
             ---> time

block_idx:   |1 2 3 4 5 6 7 0 0 0 0 0 0|
target: [-1, 4, 3, 1, 3, 4, 6, 3]
```

In this example, the block_idx contains multiple blocks (1 to 7) and the target vector contains a target for each block (e.g. target for block 4 is given by target[4] = 3).

With this method, we can efficiently collect the per-block predictions with torch.scatter_reduce and thereby ignore the padded time steps.




This work is licensed under a
[Creative Commons Attribution-ShareAlike 4.0 International License][cc-by-sa].

[![CC BY-SA 4.0][cc-by-sa-image]][cc-by-sa]

[cc-by-sa]: http://creativecommons.org/licenses/by-sa/4.0/
[cc-by-sa-image]: https://licensebuttons.net/l/by-sa/4.0/88x31.png
[cc-by-sa-shield]: https://img.shields.io/badge/License-CC%20BY--SA%204.0-lightgrey.svg
