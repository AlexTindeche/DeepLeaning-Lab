# Deep learning project involving the distilation of a motion prediction model


Model 1:
    target:
    _target_: src.model.trainer_forecast.Trainer
    dim: 32
    historical_steps: 50
    future_steps: 60
    encoder_depth: 2
    num_heads: 4
    mlp_ratio: 2.0
    qkv_bias: False
    drop_path: 0.2
    pretrained_weights: ${pretrained_weights}
    lr: ${lr}
    weight_decay: ${weight_decay}
    epochs: ${epochs}
    warmup_epochs: ${warmup_epochs}
    decoder: 
        _target_: src.model.decoder.mlp_decoder.MLPDecoder
        embed_dim: 64
        num_modes: 3
        hidden_dim: 128

    Stats:
        Run summary:
        wandb:                       epoch 59
        wandb:                lr-AdamW/pg1 0.0
        wandb:                lr-AdamW/pg2 0.0
        wandb:        train/cls_loss_epoch 1.63718
        wandb:         train/cls_loss_step 1.58762
        wandb:            train/loss_epoch 2.70447
        wandb:             train/loss_step 2.68194
        wandb: train/others_reg_loss_epoch 0.64526
        wandb:  train/others_reg_loss_step 0.68104
        wandb:        train/reg_loss_epoch 0.42203
        wandb:         train/reg_loss_step 0.41327
        wandb:         trainer/global_step 18719
        wandb:                val/reg_loss 0.41883
        wandb:                      val_MR 0.378
        wandb:           val_brier-minFDE6 3.00784
        wandb:                 val_minADE1 2.68712
        wandb:                 val_minADE6 1.12196
        wandb:                 val_minFDE1 6.80193
        wandb:                 val_minFDE6 2.38476

        ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
        ┃      Validate metric      ┃       DataLoader 0        ┃
        ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
        │       val/reg_loss        │    0.4139922857284546     │
        │          val_MR           │    0.38398003578186035    │
        │     val_brier-minFDE6     │    3.0106587409973145     │
        │        val_minADE1        │    2.7130656242370605     │
        │        val_minADE6        │    1.1117936372756958     │
        │        val_minFDE1        │     6.899008274078369     │
        │        val_minFDE6        │    2.3849031925201416     │
        └───────────────────────────┴───────────────────────────┘