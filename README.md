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


This work is licensed under a
[Creative Commons Attribution-ShareAlike 4.0 International License][cc-by-sa].

[![CC BY-SA 4.0][cc-by-sa-image]][cc-by-sa]

[cc-by-sa]: http://creativecommons.org/licenses/by-sa/4.0/
[cc-by-sa-image]: https://licensebuttons.net/l/by-sa/4.0/88x31.png
[cc-by-sa-shield]: https://img.shields.io/badge/License-CC%20BY--SA%204.0-lightgrey.svg
