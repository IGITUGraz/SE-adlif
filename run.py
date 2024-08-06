import pytorch_lightning as pl
import hydra
from omegaconf import DictConfig
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from models.pl_module import MLPSNN

# Main entry point. We use Hydra (https://hydra.cc) for configuration management. Note, that Hydra changes the working directory, such that each run gets a unique directory.

@hydra.main(config_path="config", config_name="main", version_base=None)
def main(cfg: DictConfig):
    pl.seed_everything(cfg.random_seed, workers=True)

    datamodule = hydra.utils.instantiate(cfg.dataset)
    model = MLPSNN(cfg)
    callbacks = []
    model_ckpt_tracker: ModelCheckpoint = ModelCheckpoint(
        monitor="val_acc_epoch",
        mode="max",
        save_last=False,
        save_top_k=1,
        dirpath="ckpt"
    )
    lr_monitor = LearningRateMonitor(
        logging_interval='step'
    )
    callbacks = [model_ckpt_tracker, lr_monitor]

    trainer: pl.Trainer = pl.Trainer(
        callbacks=callbacks,
        max_epochs=cfg.n_epochs,
        gradient_clip_val=1.5,
        enable_progress_bar=True,
        accelerator="auto"
        )

    trainer.fit(model, datamodule=datamodule)
    trainer.test(model, ckpt_path="best", datamodule=datamodule)

    return trainer.checkpoint_callback.best_model_score.cpu().detach().numpy()


if __name__ == "__main__":
    main()
