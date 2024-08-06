import pytorch_lightning as pl
import torch
import torchmetrics
from pytorch_lightning.utilities import grad_norm
from torch.nn import CrossEntropyLoss
from torch.optim.optimizer import Optimizer

from functional.loss import (
    calculate_weight_decay_loss,
    get_per_layer_spike_probs,
    snn_regularization,
)
from models.model import CustomSeq, resolve_model_config


class MLPSNN(pl.LightningModule):
    def __init__(
        self,
        model_config: dict,
        batch_size: int = 64,
        learning_rate: float = 1e-2,
        target_rate: list[int] = [0.01, 0.4],
        weight_decay: float = 0.01,
        monitored_metric: str = "val_acc_epoch",
        lr_mode: str = "max",
        output_func: str = "softmax",
    ) -> None:
        super().__init__()
        self.learning_rate = learning_rate
        self.monitored_metric = monitored_metric
        self.lr_mode = lr_mode
        self.target_rate = target_rate
        self.batch_size = batch_size
        self.processed_batch = 0
        self.weight_decay = weight_decay
        self.ignore_target_idx = -1
        self.output_func = output_func
        

        layers = resolve_model_config(model_config=model_config)
        self.model = CustomSeq(
            *layers,
        )
        self.output_size = self.model.output_size
        # optimizer config

        metrics = torchmetrics.MetricCollection(
            {
                "acc": torchmetrics.Accuracy(
                    task="multiclass",  # type: ignore
                    num_classes=self.output_size,
                    average="micro",
                    ignore_index=self.ignore_target_idx,
                )
            }
        )
        self.train_metric = metrics.clone(prefix="train_")
        self.val_metric = metrics.clone(prefix="val_")
        self.test_metric = metrics.clone(prefix="test_")
        self.loss = CrossEntropyLoss(ignore_index=self.ignore_target_idx)

        self.save_hyperparameters()

    # cf. https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.core.LightningModule.html#lightning.pytorch.core.LightningModule.configure_optimizers
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(params=self.parameters(), lr=self.learning_rate)

        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer,
            mode=self.lr_mode,
            factor=0.9,
            patience=3,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "monitor": self.monitored_metric,
            },
        }

    def forward(
        self, inputs: tuple[torch.Tensor, ...]
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        return self.model(inputs)

    def on_before_optimizer_step(self, optimizer: Optimizer) -> None:
        # log weights gradient norm
        self.log_dict(grad_norm(self, norm_type=2))

    def on_train_batch_end(self, outputs, batch, batch_idx: int):
        self.model.apply_parameter_constraints()

    def process_predictions_and_compute_losses(
        self, outputs, targets, block_idx
    ):
        """
        Process the model output into prediction
        with respect to the temporal segmentation defined by the
        block_idx tensor.
        Then compute losses
        Args:
            outputs (torch.Tensor): full outputs
            targets (torch.Tensor): targets
            block_idx (torch.Tensor): tensor of index that determined which temporal segements of
            output time-step depends on which specific target,
            used by the scatter reduce operation.

        Returns:
            (): _description_
        """
        # compute softmax for every time-steps with respect to
        # the number of class
        if self.output_func == "softmax":
            outputs = torch.softmax(outputs, -1)
        # create a zero array of size (batch, number_of_targets, number_of_classes)
        # this will be used to defined the prediction for each targets for each classes
        block_outputs = torch.zeros(
            size=(targets.size(0), targets.size(1), outputs.size(2)),
            dtype=outputs.dtype,
            device=outputs.device,
        )
        block_idx = block_idx.unsqueeze(-1)
        
        block_output = torch.scatter_reduce(
            block_outputs,
            dim=1,
            index=block_idx.broadcast_to(outputs.shape),
            src=outputs,
            reduce="sum",
            include_self=False,
        )

        outputs_reduce = block_output.reshape(-1, outputs.size(-1))
        targets_reduce = targets.flatten()

        loss = self.loss(outputs_reduce.float(), targets_reduce)
        return (outputs_reduce, loss, block_idx)

    def update_and_log_metrics(
        self,
        outputs: torch.Tensor,
        targets: torch.Tensor,
        loss: float,
        metrics: torchmetrics.MetricCollection,
        aux_losses: dict,
        prefix: str,
    ):
        """
        Method centralizing the metrics logging mecanisms.

        Args:
            outputs_reduce (torch.Tensor): output prediction
            targets_reduce (torch.Tensor): target
            loss (float): loss
            metrics (torchmetrics.MetricCollection): collection of torchmetrics metrics
            aux_metrics (dict): auxiliary metrics that do not
            fit the torchmetrics logic
            prefix (str): prefix defining the stage of model either
            "train_": training stage
            "val_": validation stage
            "test_": testing stage
            Those prefix prevent clash of names in the logger.

        """

        metrics(outputs, targets)
        self.log_dict(
            metrics,
            prog_bar=True,
            on_epoch=True,
            on_step=True,
            batch_size=self.batch_size,
        )
        self.log(
            f"{prefix}loss",
            loss,
            prog_bar=True,
            on_epoch=True,
            on_step=True,
            batch_size=self.batch_size,
        )
        for k, v in aux_losses.items():
            self.log(
                f"{prefix}{k}",
                v,
                prog_bar=True,
                on_epoch=True,
                on_step=True,
                batch_size=self.batch_size,
            )

    def training_step(self, batch, batch_idx):
        inputs, targets, block_idx = batch
        outputs = self(inputs,)
        (
            outputs_reduce,
            loss,
            block_idx,
        ) = self.process_predictions_and_compute_losses(
            outputs, targets, block_idx
        )

        weight_decay_loss = self.weight_decay * calculate_weight_decay_loss(self.model)

        aux_metrics = {
            "weight_decay_loss": weight_decay_loss,
        }
        targets_reduce = targets.flatten()
        self.update_and_log_metrics(
            outputs_reduce,
            targets_reduce,
            loss,
            self.train_metric,
            aux_metrics,
            prefix="train_",
        )

        return loss + weight_decay_loss

    def validation_step(self, batch, batch_idx):
        inputs, targets, block_idx = batch
        outputs = self(inputs)

        (
            outputs_reduce,
            loss,
            block_idx,
        ) = self.process_predictions_and_compute_losses(
            outputs, targets, block_idx
        )
        targets_reduce = targets.flatten()

        self.update_and_log_metrics(
            outputs_reduce,
            targets_reduce,
            loss,
            self.val_metric,
            {},
            prefix="val_",
        )
        # report statistics (weights and spiking distribution) and plot an example of
        # model behavior against a random input
       
        return loss

    def test_step(self, batch, batch_idx):
        inputs, targets, block_idx = batch
        outputs = self(inputs)

        (
            outputs_reduce,
            loss,
            block_idx,
        ) = self.process_predictions_and_compute_losses(
            outputs, targets, block_idx
        )

        targets_reduce = targets.flatten()
        self.update_and_log_metrics(
            outputs_reduce,
            targets_reduce,
            loss,
            self.test_metric,
            {},
            prefix="test_",
        )

        return loss
