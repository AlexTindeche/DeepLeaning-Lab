# Deep learning project involving the distilation of a motion prediction model

The project is ordered into 3 parts, as presented in the poster:
1. CRD can be found on dev-alex
2. RKD can be found on dev-marius
3. HKD can be found on AlexP-Feature-based-Distillation 

To run the training on brances `dev-alex` and `dev-marius` use the CLI command: 
```bash
python run_distil.py data_root="./data_root" gpus=1 limit_train_batches=0.1 limit_val_batches=0.2 monitor=val_minFDE6 wandb=online epochs=21
```
