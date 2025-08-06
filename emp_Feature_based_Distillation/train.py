import os

import hydra
import pytorch_lightning as pl
import torch
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from pytorch_lightning.callbacks import (LearningRateMonitor, ModelCheckpoint,
                                         RichModelSummary, RichProgressBar)
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger


# @hydra.main(version_base=None, config_path="conf", config_name="config")
# def main(conf):
#     torch.use_deterministic_algorithms(True)
#     torch.multiprocessing.set_start_method("spawn")
#     pl.seed_everything(conf.seed, workers=True)
#     torch.backends.cudnn.deterministic = True
#     output_dir = HydraConfig.get().runtime.output_dir

#     if conf.wandb != "disable":
#         logger = WandbLogger(
#             project="EMP",
#             name=conf.output,
#             mode=conf.wandb,
#             log_model="all",
#             resume=conf.checkpoint is not None,
#         )
#     else:
#         logger = TensorBoardLogger(save_dir=output_dir, name="logs")

#     callbacks = [
#         ModelCheckpoint(
#             dirpath=os.path.join(output_dir, "checkpoints"),
#             filename="{epoch}",
#             monitor=f"{conf.monitor}",
#             mode="min",
#             save_top_k=conf.save_top_k,
#             save_last=True,
#         ),
#         RichModelSummary(max_depth=1),
#         RichProgressBar(),
#         LearningRateMonitor(logging_interval="epoch"),
#     ]

#     trainer = pl.Trainer(
#         logger=logger,
#         gradient_clip_val=conf.gradient_clip_val,
#         gradient_clip_algorithm=conf.gradient_clip_algorithm,
#         max_epochs=conf.epochs,
#         accelerator="mps",
#         devices=1,
#         strategy="ddp_find_unused_parameters_false" if conf.gpus > 1 else None,
#         callbacks=callbacks,
#         limit_train_batches=conf.limit_train_batches,
#         limit_val_batches=conf.limit_val_batches,
#         sync_batchnorm=conf.sync_bn,
#     )

#     model = instantiate(conf.model.target)
#     datamodule = instantiate(conf.datamodule)

#     trainer.fit(model, datamodule, ckpt_path=conf.checkpoint)

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(conf):
    torch.use_deterministic_algorithms(True)
    torch.multiprocessing.set_start_method("spawn")
    pl.seed_everything(conf.seed, workers=True)
    torch.backends.cudnn.deterministic = True
    output_dir = HydraConfig.get().runtime.output_dir

    if conf.wandb != "disable":
        logger = WandbLogger(
            project="EMP",
            name=conf.output,
            mode=conf.wandb,
            log_model="all",
            resume=conf.checkpoint is not None,
        )
    else:
        logger = TensorBoardLogger(save_dir=output_dir, name="logs")

    callbacks = [
        ModelCheckpoint(
            dirpath=os.path.join(output_dir, "checkpoints"),
            filename="{epoch}",
            monitor=f"{conf.monitor}",
            mode="min",
            save_top_k=conf.save_top_k,
            save_last=True,
        ),
        RichModelSummary(max_depth=1),
        RichProgressBar(),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    if conf.isHintDistil == "enabled":
        print("\nStarting hint based distilation training\n")
        model = instantiate(conf.model.target_hint_distill)
        datamodule = instantiate(conf.datamodule)

        # hint training 
        hint_epochs = conf.hint_epochs if hasattr(conf, "hint_epochs") else conf.epochs // 5

        trainer_hint = pl.Trainer(
            logger=logger,
            gradient_clip_val=conf.gradient_clip_val,
            gradient_clip_algorithm=conf.gradient_clip_algorithm,
            max_epochs=hint_epochs,
            accelerator="mps",
            devices=1,
            strategy="ddp_find_unused_parameters_false" if conf.gpus > 1 else None,
            callbacks=callbacks,
            limit_train_batches=conf.limit_train_batches,
            limit_val_batches=conf.limit_val_batches,
            sync_batchnorm=conf.sync_bn,
        )

        print(f"begin stage 1, hint pretraining for {hint_epochs} epochs")
        trainer_hint.fit(model, datamodule, ckpt_path=conf.checkpoint)

        
        print("switching to full training stage")
        model.switch_to_main_training()

        # full model training 
        trainer_main = pl.Trainer(
            logger=logger,
            gradient_clip_val=conf.gradient_clip_val,
            gradient_clip_algorithm=conf.gradient_clip_algorithm,
            max_epochs=conf.epochs,
            accelerator="mps",
            devices=1,
            strategy="ddp_find_unused_parameters_false" if conf.gpus > 1 else None,
            callbacks=callbacks,
            limit_train_batches=conf.limit_train_batches,
            limit_val_batches=conf.limit_val_batches,
            sync_batchnorm=conf.sync_bn,
        )

        print(f"start stage 2, full training for {conf.epochs} epochs")
        trainer_main.fit(model, datamodule)
    elif conf.isHintDistil == "disabled":
        print("\nStarting normal training\n")
        trainer = pl.Trainer(
            logger=logger,
            gradient_clip_val=conf.gradient_clip_val,
            gradient_clip_algorithm=conf.gradient_clip_algorithm,
            max_epochs=conf.epochs,
            accelerator="mps",
            devices=1,
            strategy="ddp_find_unused_parameters_false" if conf.gpus > 1 else None,
            callbacks=callbacks,
            limit_train_batches=conf.limit_train_batches,
            limit_val_batches=conf.limit_val_batches,
            sync_batchnorm=conf.sync_bn,
        )

        model = instantiate(conf.model.target_normal_training)
        datamodule = instantiate(conf.datamodule)

        trainer.fit(model, datamodule, ckpt_path=conf.checkpoint)
    else:
        raise Exception("IsHintDistil in config should be either enabled or disabled")



if __name__ == "__main__":
    main()
