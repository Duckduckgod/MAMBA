# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import logging
import os
import torch
from mamba_core.utils.c2_model_loading import load_c2_format
from mamba_core.utils.imports import import_file
from mamba_core.utils.model_serialization import load_state_dict
from mamba_core.utils.model_zoo import cache_url


class Checkpointer:
    def __init__(
        self,
        model,
        optimizer=None,
        scheduler=None,
        save_dir="",
        save_to_disk=None,
        logger=None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.save_dir = save_dir
        self.save_to_disk = save_to_disk
        if logger is None:
            logger = logging.getLogger(__name__)
        self.logger = logger

    def save(self, name, **kwargs):
        if not self.save_dir:
            return

        if not self.save_to_disk:
            return

        data = {}
        data["model"] = self.model.state_dict()
        if self.optimizer is not None:
            data["optimizer"] = self.optimizer.state_dict()
        if self.scheduler is not None:
            data["scheduler"] = self.scheduler.state_dict()
        data.update(kwargs)

        save_file = os.path.join(self.save_dir, f"{name}.pth")
        self.logger.info(f"Saving checkpoint to {save_file}")
        torch.save(data, save_file)
        self.tag_last_checkpoint(save_file)

    def save_pure_model(self, name):
        if not self.save_dir:
            return

        if not self.save_to_disk:
            return

        model_params = self.model.state_dict()
        save_file = os.path.join(self.save_dir, f"{name}.pth")
        self.logger.info(f"Saving pure model to {save_file}")
        torch.save(model_params, save_file)

    def load(self, f=None, use_latest=True, ignore=False, flownet=False):
        if self.has_checkpoint() and use_latest:
            # override argument with existing checkpoint
            f = self.get_checkpoint_file()
        if not f:
            # no checkpoint could be found
            self.logger.info("No checkpoint found. Initializing model from scratch")
            return {}
        self.logger.info(f"Loading checkpoint from {f}")
        checkpoint = self._load_file(f)
        self._load_model(checkpoint, flownet)
        if ignore:
            checkpoint.pop("optimizer")
            checkpoint.pop("scheduler")
        else:
            if "optimizer" in checkpoint and self.optimizer:
                self.logger.info(f"Loading optimizer from {f}")
                self.optimizer.load_state_dict(checkpoint.pop("optimizer"))
            if "scheduler" in checkpoint and self.scheduler:
                self.logger.info(f"Loading scheduler from {f}")
                self.scheduler.load_state_dict(checkpoint.pop("scheduler"))

        # return any further checkpoint data
        return checkpoint

    def load_flownet(self, f=None):
        self.logger.info(f"Loading flownet from {f}")
        loaded_state_dict = torch.load(f)["state_dict"]
        load_state_dict(self.model, loaded_state_dict, flownet=True)

    def has_checkpoint(self):
        save_file = os.path.join(self.save_dir, "last_checkpoint")
        return os.path.exists(save_file)

    def get_checkpoint_file(self):
        save_file = os.path.join(self.save_dir, "last_checkpoint")
        try:
            with open(save_file) as f:
                last_saved = f.read()
                last_saved = last_saved.strip()
        except OSError:
            # if file doesn't exist, maybe because it has just been
            # deleted by a separate process
            last_saved = ""
        return last_saved

    def tag_last_checkpoint(self, last_filename):
        save_file = os.path.join(self.save_dir, "last_checkpoint")
        with open(save_file, "w") as f:
            f.write(last_filename)

    def _load_file(self, f):
        return torch.load(f, map_location=torch.device("cpu"))

    def _load_model(self, checkpoint, flownet=False):
        load_state_dict(self.model, checkpoint.pop("model"), flownet=flownet)


class DetectronCheckpointer(Checkpointer):
    def __init__(
        self,
        cfg,
        model,
        optimizer=None,
        scheduler=None,
        save_dir="",
        save_to_disk=None,
        logger=None,
    ):
        super().__init__(model, optimizer, scheduler, save_dir, save_to_disk, logger)
        self.cfg = cfg.clone()

    def _load_file(self, f):
        # catalog lookup
        if f.startswith("catalog://"):
            paths_catalog = import_file(
                "mamba_core.config.paths_catalog", self.cfg.PATHS_CATALOG, True
            )
            catalog_f = paths_catalog.ModelCatalog.get(f[len("catalog://") :])
            self.logger.info(f"{f} points to {catalog_f}")
            f = catalog_f
        # download url files
        if f.startswith("http"):
            # if the file is a url path, download it and cache it
            cached_f = cache_url(f)
            self.logger.info(f"url {f} cached in {cached_f}")
            f = cached_f
        # convert Caffe2 checkpoint from pkl
        if f.endswith(".pkl"):
            return load_c2_format(self.cfg, f)
        # load native detectron.pytorch checkpoint
        loaded = super()._load_file(f)
        if "model" not in loaded:
            loaded = dict(model=loaded)
        return loaded
