# Deep learning project involving the distilation of a motion prediction model


Modifications:
1. Added support for different attention types in the model configuration.  
    For this I added the following:  
    a. In transformer_blocks.py I added 2 classes:  
        - LinearAttentionBlock  
        - PerformerAttentionBlock

Model 1:  

    params: 305K

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

    