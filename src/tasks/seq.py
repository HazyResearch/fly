from typing import Any, List

import torch
import hydra
from pytorch_lightning import LightningModule, LightningDataModule
from torchmetrics import MetricCollection

from einops import rearrange

from src.utils.utils import get_logger
from src.utils.optim import group_parameters_for_optimizer

log = get_logger(__name__)


class SequenceModel(LightningModule):

    def __init__(self, cfg, model_cfg=None):
        """If model_cfg is passed, it will take precedence over cfg.model
        """
        super().__init__()
        # this line ensures params passed to LightningModule will be saved to ckpt
        # it also allows to access params with 'self.hparams' attribute
        self.save_hyperparameters(cfg)
        self.cfg = cfg
        self.model_cfg = model_cfg or self.cfg.model

        # Init lightning datamodule
        log.info(f"Instantiating datamodule <{cfg.datamodule._target_}>")
        # Calling this self.datamodule will mess with PL since it also assigns self.datamodule
        self._datamodule: LightningDataModule = hydra.utils.instantiate(cfg.datamodule)
        self._datamodule.prepare_data()
        self._datamodule.setup()

        if hasattr(self._datamodule, 'num_classes'):
            self.model_cfg.num_classes = self._datamodule.num_classes
        if (hasattr(self._datamodule, 'vocab_size')
            and self.model_cfg.get('embedding_cfg', None) is not None):
            self.model_cfg.embedding_cfg.num_embeddings = self._datamodule.vocab_size
        log.info(f"Instantiating model <{self.model_cfg._target_}>")
        self.model = hydra.utils.instantiate(self.model_cfg, _recursive_=False)

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
        if 'eval' in self.cfg and 'metrics' in self.cfg.eval:
            metrics_cfg = self.cfg.eval.metrics
        else:
            metrics_cfg = {'acc': {'_target_': 'torchmetrics.Accuracy'}}
        metrics = MetricCollection({name: hydra.utils.instantiate(cfg)
                                    for name, cfg in metrics_cfg.items()})
        self.train_metrics = metrics.clone(prefix='train/')
        self.val_metrics = metrics.clone(prefix='val/')
        self.test_metrics = metrics.clone(prefix='test/')

    def setup(self, stage=None):
        # Init model, but not if we're testing
        if stage == 'test':
            return
        if hasattr(self._datamodule, 'num_classes'):
            self.model_cfg.num_classes = self._datamodule.num_classes
        if (hasattr(self._datamodule, 'vocab_size')
            and self.model_cfg.get('embedding_cfg', None) is not None):
            self.model_cfg.embedding_cfg.num_embeddings = self._datamodule.vocab_size

        # BC init model only when it's None
        if self.model is None:
            log.info(f"Instantiating model <{self.model_cfg._target_}>")
            self.model = hydra.utils.instantiate(self.model_cfg, _recursive_=False)
        else:
            log.info("Model Already initialized")

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def step(self, batch: Any, is_train=True):
        try:
            x, y, lengths = batch
        except ValueError:
            x, y = batch
            lengths = None
        if is_train and self.mixup is not None:
            x, y = self.mixup(x, y)
        targets = y.argmax(dim=-1) if is_train and self.mixup is not None else y  # In case of Mixup
        output = self.forward(x) if lengths is None else self.forward(x, lengths=lengths)
        loss = self.loss_fn(output, y) if is_train else self.loss_fn_val(output, y)
        return loss, output, targets

    def shared_step(self, batch: Any, batch_idx: int, phase='train'):
        loss, output, targets = self.step(batch, is_train=(phase == 'train'))
        metrics = getattr(self, f'{phase}_metrics')(output, targets)
        self.log(f"{phase}/loss", loss, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return {"loss": loss, "output": output, "targets": targets}

    def training_step(self, batch: Any, batch_idx: int):
        return self.shared_step(batch, batch_idx, phase='train')

    def validation_step(self, batch: Any, batch_idx: int):
        return self.shared_step(batch, batch_idx, phase='val')

    def test_step(self, batch: Any, batch_idx: int):
        return self.shared_step(batch, batch_idx, phase='test')

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
                                 'interval': self.cfg.train.get('scheduler_interval', 'step'),
                                 'monitor': self.cfg.train.get('scheduler_monitor', 'val/loss')}


class SequenceDualModel(SequenceModel):

    def step(self, batch: Any, is_train=True):
        x1, x2, y, lengths1, lengths2 = batch
        output = self.forward(x1, x2, lengths1=lengths1, lengths2=lengths2)
        loss = self.loss_fn(output, y) if is_train else self.loss_fn_val(output, y)
        output = torch.argmax(output, dim=1)
        return loss, output, y


class SequenceLMModel(SequenceModel):

    def __init__(self, cfg, model_cfg=None):
        """If model_cfg is passed, it will take precedence over cfg.model
        """
        # For some reason calling LightningModule.__init__(self) causes error when I call
        # self.save_hyperparameters(cfg)
        super(SequenceModel, self).__init__()
        # this line ensures params passed to LightningModule will be saved to ckpt
        # it also allows to access params with 'self.hparams' attribute
        self.save_hyperparameters(cfg)
        self.cfg = cfg
        self.model_cfg = model_cfg or self.cfg.model

        # Init lightning datamodule
        log.info(f"Instantiating datamodule <{cfg.datamodule._target_}>")
        # Calling this self.datamodule will mess with PL since it also assigns self.datamodule
        self._datamodule: LightningDataModule = hydra.utils.instantiate(cfg.datamodule)
        self._datamodule.prepare_data()
        self._datamodule.setup()

        if hasattr(self._datamodule, 'num_classes'):
            self.model_cfg.num_classes = self._datamodule.num_classes
        if (hasattr(self._datamodule, 'vocab_size')
            and self.model_cfg.get('embedding_cfg', None) is not None):
            self.model_cfg.embedding_cfg.num_embeddings = self._datamodule.vocab_size
        log.info(f"Instantiating model <{self.model_cfg._target_}>")
        config = hydra.utils.instantiate(self.model_cfg.pop('config'), _recursive_=False)
        self.model = hydra.utils.instantiate(self.model_cfg, config, _recursive_=False)

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
        if 'eval' in self.cfg and 'metrics' in self.cfg.eval:
            metrics_cfg = self.cfg.eval.metrics
        else:
            metrics_cfg = {'acc': {'_target_': 'torchmetrics.Accuracy'}}
        metrics = MetricCollection({name: hydra.utils.instantiate(cfg)
                                    for name, cfg in metrics_cfg.items()})
        self.train_metrics = metrics.clone(prefix='train/')
        self.val_metrics = metrics.clone(prefix='val/')
        self.test_metrics = metrics.clone(prefix='test/')

    def setup(self, stage=None):
        # Init model, but not if we're testing
        if stage == 'test':
            return
        if hasattr(self._datamodule, 'num_classes'):
            self.model_cfg.num_classes = self._datamodule.num_classes
        if (hasattr(self._datamodule, 'vocab_size')
            and self.model_cfg.get('embedding_cfg', None) is not None):
            self.model_cfg.embedding_cfg.num_embeddings = self._datamodule.vocab_size

        # BC init model only when it's None
        if self.model is None:
            log.info(f"Instantiating model <{self.model_cfg._target_}>")
            config = hydra.utils.instantiate(self.model_cfg.pop('config'), _recursive_=False)
            self.model = hydra.utils.instantiate(self.model_cfg, config, _recursive_=False)
        else:
            log.info("Model Already initialized")

    def step(self, batch: Any, is_train=True):
        x, y = batch['input_ids'], batch['labels']
        output = self.forward(input_ids=x, attention_mask=batch['attention_mask']).logits
        # Need to shift since huggingface has input_ids == labels
        output = rearrange(output[..., :-1, :], '... C -> (...) C')
        y = rearrange(y[..., 1:], '... -> (...)')
        loss = self.loss_fn(output, y) if is_train else self.loss_fn_val(output, y)
        return loss, output, y

    def shared_step(self, batch: Any, batch_idx: int, phase='train'):
        loss, output, targets = self.step(batch, is_train=(phase == 'train'))
        metrics = getattr(self, f'{phase}_metrics')(output, targets)
        self.log(f"{phase}/loss", loss, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return {"loss": loss, "output": output, "targets": targets}
