import collections
import os
import typing

import hydra
import timm
import torch
from omegaconf import ListConfig, OmegaConf, DictConfig
from omegaconf.dictconfig import DictConfig as DC
from omegaconf.listconfig import ListConfig as LC

import pytorch_lightning as pl
from hydra.utils import instantiate, to_absolute_path
from importlib import import_module
from torch.serialization import safe_globals
import omegaconf.base
from src.model.emp import EMP
from src.model.layers.transformer_blocks import Block
from src.model.layers.transformer_blocks import Mlp
from src.model.layers.lane_embedding import LaneEmbeddingLayer
from src.model.layers.multimodal_decoder_emp import MultimodalDecoder

# Get all torch.nn classes and functions
torch_nn_globals = []

# Add all classes from torch.nn
for name in dir(torch.nn):
    attr = getattr(torch.nn, name)
    if hasattr(attr, '__module__') and hasattr(attr, '__qualname__'):
        torch_nn_globals.append(attr)

# Add all classes from torch.nn.modules (more comprehensive)
import torch.nn.modules
for module_name in dir(torch.nn.modules):
    try:
        module = getattr(torch.nn.modules, module_name)
        if hasattr(module, '__module__'):
            # Add all classes in this submodule
            for class_name in dir(module):
                cls = getattr(module, class_name)
                if hasattr(cls, '__module__') and hasattr(cls, '__qualname__'):
                    torch_nn_globals.append(cls)
    except:
        continue

torch.serialization.add_safe_globals(torch_nn_globals)

torch.serialization.add_safe_globals([omegaconf.base.ContainerMetadata,
                                    omegaconf.nodes.AnyNode,
                                    omegaconf.base.Metadata, 
                                    typing.Any, 
                                    dict, 
                                    collections.defaultdict, 
                                    list, 
                                    tuple, 
                                    str, 
                                    int, 
                                    float, 
                                    bool, 
                                    EMP,
                                    torch.nn.modules.linear.Linear,
                                    torch.nn.modules.container.ModuleList,
                                    Block,
                                    torch.nn.modules.normalization.LayerNorm,
                                    torch.nn.modules.activation.MultiheadAttention,
                                    torch.nn.modules.linear.NonDynamicallyQuantizableLinear,
                                    torch.nn.modules.linear.Identity,
                                    Mlp,
                                    torch.nn.modules.activation.GELU,
                                    timm.layers.drop.DropPath,
                                    LaneEmbeddingLayer,
                                    MultimodalDecoder,
                                    ])


@hydra.main(version_base=None, config_path="./conf/", config_name="config")
def main(conf):
    pl.seed_everything(conf.seed)
    
    log_base_dir = "/".join( conf.checkpoint.split("/")[:-2] ) + "/"
    checkpoint = to_absolute_path(conf.checkpoint)
    assert os.path.exists(checkpoint), f"Checkpoint {checkpoint} does not exist"

    model_path = conf.model.target._target_
    module = import_module(model_path[: model_path.rfind(".")])
    Model: pl.LightningModule = getattr(module, model_path[model_path.rfind(".") + 1 :])
    with safe_globals([DictConfig, ListConfig, OmegaConf, LC, DC]):
        print(f"Loading model from {checkpoint}")
        model = Model.load_from_checkpoint(
            checkpoint,
            )

    trainer = pl.Trainer(
        logger=False,
        accelerator="gpu",
        devices=conf.gpus,
        max_epochs=1,
        limit_val_batches=conf.limit_val_batches,
        limit_test_batches=conf.limit_test_batches,
    )

    datamodule: pl.LightningDataModule = instantiate(conf.datamodule, test=conf.test)

    if not conf.test:
        trainer.validate(model, datamodule)
    else:
        trainer.test(model, datamodule)


if __name__ == "__main__":
    main()
