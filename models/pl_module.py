import pytorch_lightning as pl
import torch
import torchmetrics
from torch.nn import CrossEntropyLoss
from models.alif import EFAdLIF, SEAdLIF
from models.li import LI
from models.lif import LIF


layer_map = {
    "lif": LIF,
    "se_adlif": SEAdLIF,
    "ef_adlif": EFAdLIF,
}


class MLPSNN(pl.LightningModule):
    def __init__(
        self,
        cfg,
    ) -> None:
        super().__init__()
        self.ignore_target_idx = -1
        self.cell = layer_map[cfg.cell]
        self.l1 = self.cell(cfg)
        self.two_layers = cfg.two_layers
        if cfg.two_layers:
            self.l2 = self.cell(cfg)
        self.out_layer = LI(cfg)
        self.output_size = cfg.dataset.num_classes
        self.init_metrics_and_loss(cfg)

    def forward(
        self, inputs: tuple[torch.Tensor, ...]
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        s1 = self.l1.get_initial_state(inputs[0].shape[0], inputs[0].device)
        s_out = self.out_layer.get_initial_state(inputs[0].shape[0], inputs[0].device)
        if self.two_layers:
            s2 = self.l2.get_initial_state(inputs[0].shape[0], inputs[0].device)
        out_sequence = []
        for t, x_t in enumerate(inputs):
            out, s1 = self.l1(x_t, s1)
            if self.two_layers:
                out, s2 = self.l2(out, s2)
            out, s_out = self.out_layer(out, s_out)
            out_sequence.append(out)
        return out_sequence

    def on_train_batch_end(self, outputs, batch, batch_idx: int):
        self.l1.apply_parameter_constraints()
        if self.two_layers:
            self.l2.apply_parameter_constraints()
        self.out_layer.apply_parameter_constraints()

    def process_predictions_and_compute_losses(self, outputs, targets, block_idx):
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

    def training_step(self, batch, batch_idx):
        inputs, targets, block_idx = batch
        outputs = self(
            inputs,
        )
        (
            outputs_reduce,
            loss,
            block_idx,
        ) = self.process_predictions_and_compute_losses(outputs, targets, block_idx)

        targets_reduce = targets.flatten()
        self.update_and_log_metrics(
            outputs_reduce,
            targets_reduce,
            loss,
            self.train_metric,
            prefix="train_",
        )

        return loss

    def validation_step(self, batch, batch_idx):
        inputs, targets, block_idx = batch
        outputs = self(inputs)
        (
            outputs_reduce,
            loss,
            block_idx,
        ) = self.process_predictions_and_compute_losses(outputs, targets, block_idx)
        targets_reduce = targets.flatten()

        self.update_and_log_metrics(
            outputs_reduce,
            targets_reduce,
            loss,
            self.val_metric,
            {},
            prefix="val_",
        )

        return loss

    def test_step(self, batch, batch_idx):
        inputs, targets, block_idx = batch
        outputs = self(inputs)

        (
            outputs_reduce,
            loss,
            block_idx,
        ) = self.process_predictions_and_compute_losses(outputs, targets, block_idx)

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

    def init_metrics_and_loss(self, cfg):
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
