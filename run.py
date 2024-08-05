import pytorch_lightning as pl
import hydra
from omegaconf import DictConfig, OmegaConf
#from datasets.number_recognition import NumberRecognitionDM
from datasets.multidigit_addition import MultiDigitAdditionDM
from datasets.temporal_gaussian_array import TemporalGaussianArrayDM
from datasets.temporal_gaussian_array_poisson import TemporalGaussianArrayPoissonDM
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from model.pl_module import MLPSNN
import sys
import traceback
import matplotlib
import copy
matplotlib.use("agg")
dataset_map = {
    "multidigit_addition": MultiDigitAdditionDM,
    "temporal_gaussian_array": TemporalGaussianArrayDM,
    "temporal_gaussian_array_poisson": TemporalGaussianArrayPoissonDM,
}
@hydra.main(config_path="config", config_name="main", version_base=None)
def main(cfg: DictConfig):
    try:
        # DictConfig to standard python dict
        cfg = OmegaConf.to_container(cfg, resolve=True)
        pl.seed_everything(cfg["seed_everything"], workers=True)

        '''
        data = MultiDigitAdditionDM(
            **cfg["data"]
            )

        cfg_val = copy.deepcopy(cfg)
        cfg_val["data"]["n_digits"] = cfg_val["data"]["n_digits"] + 5
        data_val = MultiDigitAdditionDM(
            **cfg_val["data"]
            )
        '''

        data = TemporalGaussianArrayPoissonDM(
            **cfg["data"]
        )
        data_val = TemporalGaussianArrayPoissonDM(
            **cfg["data"]
        )

        model = MLPSNN(
            **cfg["model"],
            batch_size=cfg["data"]["batch_size"],
            monitored_metric=cfg["monitored_metric"],
            lr_mode=cfg["monitor_mode"],
            log_every_n_epoch=cfg["log_every_n_epoch"],
            learning_rate=cfg["learning_rate"],
            weight_decay=cfg["weight_decay"],
            spike_reg=cfg["spike_reg"],
            target_rate=cfg["target_rate"]
            )
        callbacks = []
        model_ckpt_tracker: ModelCheckpoint = ModelCheckpoint(
            monitor=cfg["monitored_metric"],
            mode=cfg["monitor_mode"],
            save_last=False,
            save_top_k=1,
            dirpath="ckpt"
        )
        lr_monitor = LearningRateMonitor(
            logging_interval='step'
        )
        callbacks = [model_ckpt_tracker, lr_monitor]
        try: 
            from aim.pytorch_lightning import AimLogger
            
            # remove stdout 
            enable_progress_bar = True
            logger = AimLogger(
                repo= "/home/sabathiels/code/adlif/adlif-main/sim_results" + "/.aim", #cfg["result_dir"]+ "/.aim",
                experiment=cfg["experiment"],
            )
            logger = logger
        except ImportError:
            print("[W] Aim is not installed, fallback to stdout", flush=True)
            logger = True
            enable_progress_bar = True
        trainer: pl.Trainer = pl.Trainer(
            logger=logger,
            callbacks=callbacks,
            max_epochs=cfg["max_epochs"],
            gradient_clip_val=cfg["gradient_clip_val"],
            enable_progress_bar=enable_progress_bar,
            log_every_n_steps=cfg["log_every_n_epoch"],
            accelerator=cfg["accelerator"]
            )
        # add hyper-parameters not automatically tracked
        trainer.logger.log_hyperparams(cfg["data"])
        trainer.logger.log_hyperparams(
            {
                "seed": cfg["seed_everything"],
                "gradient_clip_val": cfg["gradient_clip_val"],
                
            }
        )

        trainer.fit(model, datamodule=data)
        trainer.test(model, ckpt_path="best", datamodule=data)
        trainer.logger.log_hyperparams(
            {"ckpt_path": trainer.checkpoint_callback.best_model_path}
        )
        trainer.logger.save()
        return trainer.checkpoint_callback.best_model_score.cpu().detach().numpy()
    except Exception:
        traceback.print_exc(file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
