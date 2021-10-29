from typing import Any, List

import torch
import hydra
from pytorch_lightning import LightningModule
from torchmetrics.classification.accuracy import Accuracy

from src.utils.utils import get_logger
from src.utils.optim import group_parameters_for_optimizer

log = get_logger(__name__)


class SequenceModel(LightningModule):

    def __init__(self, cfg):
        super().__init__()
        # this line ensures params passed to LightningModule will be saved to ckpt
        # it also allows to access params with 'self.hparams' attribute
        self.save_hyperparameters(cfg)
        self.cfg = cfg

        # Init lightning datamodule
        log.info(f"Instantiating datamodule <{cfg.datamodule._target_}>")
        # Calling this self.datamodule will mess with PL since it also assigns self.datamodule
        self._datamodule: LightningDataModule = hydra.utils.instantiate(cfg.datamodule)
        self._datamodule.prepare_data()
        self._datamodule.setup()

        if hasattr(self._datamodule, 'num_classes'):
            self.cfg.model.num_classes = self._datamodule.num_classes
        if hasattr(self._datamodule, 'vocab_size') and self.cfg.model.embedding_cfg is not None:
            self.cfg.model.embedding_cfg.num_embeddings = self._datamodule.vocab_size
        log.info(f"Instantiating model <{self.cfg.model._target_}>")
        self.model = hydra.utils.instantiate(self.cfg.model, _recursive_=False)

        # Mixup / Cutmix
        if hasattr(self.cfg.train, 'mixup'):
            if hasattr(self._datamodule, 'num_classes'):
                self.cfg.train.mixup.num_classes = self._datamodule.num_classes
            self.mixup = hydra.utils.instantiate(self.cfg.train.mixup)
        else:
            self.mixup = None

        # loss function
        loss_fn_cfg = self.cfg.train.get('loss_fn', {'_target_': 'torch.nn.CrossEntropyLoss'})
        self.loss_fn = hydra.utils.instantiate(loss_fn_cfg)
        loss_fn_val_cfg = self.cfg.train.get('loss_fn_val', loss_fn_cfg)
        self.loss_fn_val = hydra.utils.instantiate(loss_fn_val_cfg)

        # use separate metric instance for train, val and test step
        # to ensure a proper reduction over the epoch
        self.train_accuracy = Accuracy()
        self.val_accuracy = Accuracy()
        self.test_accuracy = Accuracy()

    def setup(self, stage=None):
        # Init model, but not if we're testing
        if stage == 'test':
            return
        if hasattr(self._datamodule, 'num_classes'):
            self.cfg.model.num_classes = self._datamodule.num_classes
        if hasattr(self._datamodule, 'vocab_size') and self.cfg.model.embedding_cfg is not None:
            self.cfg.model.embedding_cfg.num_embeddings = self._datamodule.vocab_size

        # BC init model only when it's None
        if self.model is None:
            log.info(f"Instantiating model <{self.cfg.model._target_}>")
            self.model = hydra.utils.instantiate(self.cfg.model, _recursive_=False)
        else:
            log.info("Model Already initialized")

    def forward(self, x: torch.Tensor, *args, **kwargs):
        return self.model(x, *args, **kwargs)

    def step(self, batch: Any, is_train=True):
        try:
            x, y, lengths = batch
        except ValueError:
            x, y = batch
            lengths = None
        y_og = y
        if is_train and self.mixup is not None:
            x, y = self.mixup(x, y)
        logits = self.forward(x) if lengths is None else self.forward(x, lengths=lengths)
        loss = self.loss_fn(logits, y) if is_train else self.loss_fn_val(logits, y)
        preds = torch.argmax(logits, dim=1)
        return loss, preds, y_og

    def training_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)
        acc = self.train_accuracy(preds, targets)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log("train/acc", acc, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return {"loss": loss, "preds": preds, "targets": targets}

    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch, is_train=False)
        acc = self.val_accuracy(preds, targets)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log("val/acc", acc, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return {"loss": loss, "preds": preds, "targets": targets}

    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch, is_train=False)
        acc = self.test_accuracy(preds, targets)
        self.log("test/loss", loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log("test/acc", acc, on_step=False, on_epoch=True, sync_dist=True)
        return {"loss": loss, "preds": preds, "targets": targets}

    def configure_optimizers(self):
        if 'optimizer_param_grouping' in self.cfg.train:  # Set zero weight decay for some params
            parameters = group_parameters_for_optimizer(self.model, self.cfg.train.optimizer,
                                                        **self.cfg.train.optimizer_param_grouping)
        else:
            # parameters = self.model.parameters()
            parameters = self.parameters() # [21-09-08] AG: this will train task specific parameters such as Retrieval head for AAN
        optimizer = hydra.utils.instantiate(self.cfg.train.optimizer, parameters)
        if 'scheduler' not in self.cfg.train:
            return optimizer
        else:
            # lr_scheduler should be called either every step (default) or every epoch
            lr_scheduler = hydra.utils.instantiate(self.cfg.train.scheduler, optimizer)
            return [optimizer], {'scheduler': lr_scheduler,
                                 'interval': self.cfg.train.get('scheduler_interval', 'step')}
