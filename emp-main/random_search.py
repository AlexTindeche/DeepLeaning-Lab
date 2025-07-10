import os
import random
import json
from datetime import datetime
from pathlib import Path
import numpy as np

import hydra
import pytorch_lightning as pl
import torch
import wandb
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from pytorch_lightning.callbacks import (LearningRateMonitor, ModelCheckpoint,
                                         RichModelSummary, RichProgressBar, Callback)
from pytorch_lightning.loggers import WandbLogger
from omegaconf import DictConfig, OmegaConf


class EpochResultsLogger(Callback):
    """Custom callback to log results after each epoch"""
    
    def __init__(self, results_file, experiment_id):
        self.results_file = results_file
        self.experiment_id = experiment_id
        self.epoch_results = []
        
    def on_validation_epoch_end(self, trainer, pl_module):
        """Log metrics after each validation epoch"""
        epoch = trainer.current_epoch
        
        # Get current metrics
        metrics = trainer.callback_metrics
        
        # Store epoch results with safe conversion
        epoch_data = {
            "epoch": epoch + 1,
            "metrics": {}
        }
        
        for k, v in metrics.items():
            if isinstance(v, torch.Tensor):
                try:
                    val = float(v)
                    # Check for invalid values
                    if np.isnan(val) or np.isinf(val):
                        epoch_data["metrics"][k] = "invalid"
                    else:
                        epoch_data["metrics"][k] = val
                except (ValueError, TypeError, ZeroDivisionError):
                    epoch_data["metrics"][k] = "error"
            else:
                epoch_data["metrics"][k] = str(v)
        
        self.epoch_results.append(epoch_data)
        
        # Write to file immediately
        self._write_epoch_results()
        
    def _write_epoch_results(self):
        """Write current epoch results to file"""
        try:
            with open(self.results_file, 'a') as f:
                epoch_data = self.epoch_results[-1]
                f.write(f"Epoch {epoch_data['epoch']}:\n")
                
                metrics = epoch_data['metrics']
                
                # Write metrics for this epoch
                for metric_name, metric_value in metrics.items():
                    if isinstance(metric_value, (int, float)):
                        f.write(f"  {metric_name}: {metric_value:.4f}\n")
                    else:
                        f.write(f"  {metric_name}: {metric_value}\n")
                
                f.write("\n")
                f.flush()  # Ensure it's written immediately
        except Exception as e:
            print(f"Error writing epoch results: {e}")


class RandomSearchConfig:
    """Define the search space for hyperparameters"""
    
    @staticmethod
    def sample_config():
        """Sample a random configuration"""
        return {
            # Model architecture
            "dim": random.choice([24, 32, 48, 64, 96, 128]),
            "encoder_depth": random.choice([1, 2, 3, 4, 5]),
            "num_heads": random.choice([2, 4, 6, 8]),
            "mlp_ratio": random.choice([1, 2, 3, 4, 5, 6]),
            "attention_type": random.choice(["standard", "linear", "performer"]),
            
            # Decoder configuration
            "decoder_embed_dim": random.choice([32, 48, 64, 96, 128]),
            "decoder_num_modes": random.choice([3, 6, 9]),
            "decoder_hidden_dim": random.choice([64, 96, 128, 192, 256]),
        
        }
    
    @staticmethod
    def validate_config(config):
        """Ensure configuration is valid"""
        # Ensure num_heads divides dim
        if config["dim"] % config["num_heads"] != 0:
            # Adjust num_heads to be compatible
            valid_heads = [h for h in [2, 4, 6, 8] if config["dim"] % h == 0]
            if valid_heads:
                config["num_heads"] = random.choice(valid_heads)
            else:
                config["num_heads"] = 2
        
        # Ensure decoder_embed_dim is reasonable
        if config["decoder_embed_dim"] > config["dim"] * 2:
            config["decoder_embed_dim"] = config["dim"]
            
        return config


def write_experiment_header(results_file, experiment_id, search_config, model_params=None):
    """Write experiment header to results file"""
    try:
        with open(results_file, 'a') as f:
            f.write(f"Experiment {experiment_id}\n")
            f.write("-" * 50 + "\n")
            
            # Write hyperparameters
            f.write("HYPERPARAMETERS:\n")
            for key, value in search_config.items():
                if isinstance(value, float):
                    f.write(f"  {key}: {value:.4f}\n")
                else:
                    f.write(f"  {key}: {value}\n")
            
            if model_params:
                f.write(f"  model_params: {model_params:,}\n")
            
            f.write("\nTRAINING PROGRESS:\n")
            f.flush()
    except Exception as e:
        print(f"Error writing experiment header: {e}")


def write_final_results(results_file, val_metrics, status="completed"):
    """Write final validation results"""
    try:
        with open(results_file, 'a') as f:
            f.write("FINAL VALIDATION RESULTS:\n")
            if status == "failed":
                f.write("  Status: FAILED\n")
                if isinstance(val_metrics, dict) and "error" in val_metrics:
                    f.write(f"  Error: {val_metrics['error']}\n")
            else:
                for metric_name, metric_value in val_metrics.items():
                    if isinstance(metric_value, (int, float)):
                        if np.isnan(metric_value) or np.isinf(metric_value):
                            f.write(f"  {metric_name}: invalid\n")
                        else:
                            f.write(f"  {metric_name}: {metric_value:.4f}\n")
                    else:
                        f.write(f"  {metric_name}: {metric_value}\n")
            
            f.write("\n" + "="*80 + "\n\n")
            f.flush()
    except Exception as e:
        print(f"Error writing final results: {e}")


def run_single_experiment(base_conf: DictConfig, search_config: dict, experiment_id: int, results_file: str):
    """Run a single experiment with the given configuration"""
    
    # Ensure any existing wandb run is finished
    try:
        import wandb
        if wandb.run is not None:
            wandb.finish()
    except:
        pass
    
    # Create experiment name
    experiment_name = f"random_search_{experiment_id:03d}_{search_config['attention_type']}_{search_config['dim']}d"
    
    try:
        # Update configuration
        conf = OmegaConf.create(OmegaConf.to_container(base_conf))
        
        # Ensure data_root is set - this is critical for datamodule
        if conf.get("data_root") is None:
            # Try to find a valid data directory
            possible_paths = [
                "/home/alex/UNI EMERGENCY/Project/DeepLeaning-Lab/data_root",
                "/home/alex/UNI EMERGENCY/Project/DeepLeaning-Lab/data",
                "./data_root",
                "./data"
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    conf.data_root = path
                    print(f"Using data_root: {path}")
                    break
            else:
                raise ValueError(f"No valid data_root found. Tried: {possible_paths}")
        
        # Update model parameters
        for key, value in search_config.items():
            conf.model.target[key] = value
        
        # Setup output directory
        output_dir = f"./outputs/random_search/{experiment_name}"
        os.makedirs(output_dir, exist_ok=True)
        
        # Setup logger
        logger_config = {
            "experiment_id": experiment_id,
            "search_config": search_config,
            **search_config
        }
        
        logger = WandbLogger(
            project="motion-prediction-random-search",
            name=experiment_name,
            tags=["random_search", search_config["attention_type"], f"dim_{search_config['dim']}"],
            config=logger_config,
            save_dir=output_dir,
            log_model=False,
        )
        
        # Create custom callback for epoch logging
        epoch_logger = EpochResultsLogger(results_file, experiment_id)
        
        # Setup callbacks
        callbacks = [
            epoch_logger,  # Add our custom logger
            ModelCheckpoint(
                dirpath=os.path.join(output_dir, "checkpoints"),
                filename="best",
                monitor="val_minFDE6",
                mode="min",
                save_top_k=1,
                save_last=False,
            ),
            RichModelSummary(max_depth=1),
            RichProgressBar(),
            LearningRateMonitor(logging_interval="epoch"),
        ]
        
        # Setup trainer with gradient clipping to prevent instabilities
        trainer = pl.Trainer(
            logger=logger,
            max_epochs=min(conf.get("epochs", 60), 30),  # Limit epochs for random search
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices=1,
            callbacks=callbacks,
            limit_train_batches=0.1,
            limit_val_batches=0.2,
            enable_checkpointing=True,
            log_every_n_steps=50,
            gradient_clip_val=1.0,  # Add gradient clipping
            gradient_clip_algorithm="norm",
        )
        
        # Create model and datamodule
        model = instantiate(conf.model.target)
        datamodule = instantiate(conf.datamodule)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        
        # Write experiment header
        write_experiment_header(results_file, experiment_id, search_config, total_params)
        
        # Log parameters to wandb
        logger.experiment.config.update({"model_params": total_params}, allow_val_change=True)
        
        # Train model
        trainer.fit(model, datamodule)
        
        # Get final metrics with safe conversion
        val_metrics = trainer.callback_metrics
        val_metrics_dict = {}
        
        for k, v in val_metrics.items():
            if isinstance(v, torch.Tensor):
                try:
                    val = float(v)
                    if np.isnan(val) or np.isinf(val):
                        val_metrics_dict[k] = "invalid"
                    else:
                        val_metrics_dict[k] = val
                except (ValueError, TypeError, ZeroDivisionError):
                    val_metrics_dict[k] = "error"
            else:
                val_metrics_dict[k] = str(v)
        
        # Write final results
        write_final_results(results_file, val_metrics_dict, "completed")
        
        # Cleanup wandb
        if logger.experiment is not None:
            logger.experiment.finish()
        
        # Return results
        return {
            "experiment_id": experiment_id,
            "config": search_config,
            "val_metrics": val_metrics_dict,
            "model_params": total_params,
            "epochs_trained": trainer.current_epoch + 1,
            "epoch_results": epoch_logger.epoch_results
        }
        
    except Exception as e:
        print(f"Experiment {experiment_id} failed: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Write failure to file
        write_final_results(results_file, {"error": str(e)}, "failed")
        
        # Cleanup wandb
        if 'logger' in locals() and hasattr(logger, 'experiment') and logger.experiment is not None:
            logger.experiment.finish()
            
        return {
            "experiment_id": experiment_id,
            "config": search_config,
            "error": str(e),
            "failed": True
        }


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(conf: DictConfig):
    """Main random search function"""
    
    # Set seeds for reproducibility
    pl.seed_everything(2333, workers=True)
    torch.backends.cudnn.deterministic = True
    
    # Random search parameters
    num_experiments = conf.get("num_experiments", 20)
    
    print(f"Starting random search with {num_experiments} experiments")
    
    # Create results directory
    results_dir = Path("./random_search_results")
    results_dir.mkdir(exist_ok=True)
    
    # Create main results file
    results_file = results_dir / "random_search_results.txt"
    
    # Initialize results file
    with open(results_file, 'w') as f:
        f.write("RANDOM SEARCH RESULTS\n")
        f.write("=" * 80 + "\n")
        f.write(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total experiments: {num_experiments}\n")
        f.write("=" * 80 + "\n\n")
    
    # Store all results
    all_results = []
    
    # Run random search
    for i in range(num_experiments):
        print(f"\n{'='*60}")
        print(f"Experiment {i+1}/{num_experiments}")
        print(f"{'='*60}")
        
        # Sample random configuration
        search_config = RandomSearchConfig.sample_config()
        search_config = RandomSearchConfig.validate_config(search_config)
        
        print(f"Testing configuration: {search_config}")
        
        # Run experiment
        result = run_single_experiment(conf, search_config, i+1, results_file)
        all_results.append(result)
        
        # Save intermediate JSON results
        try:
            intermediate_file = results_dir / f"intermediate_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(intermediate_file, 'w') as f:
                json.dump(all_results, f, indent=2)
        except Exception as e:
            print(f"Error saving intermediate results: {e}")
    
    # Write summary at the end
    write_summary(results_file, all_results)
    
    # Save final results
    try:
        final_results_file = results_dir / f"final_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(final_results_file, 'w') as f:
            json.dump(all_results, f, indent=2)
    except Exception as e:
        print(f"Error saving final results: {e}")
    
    # Analyze results
    analyze_results(all_results)


def write_summary(results_file, all_results):
    """Write summary of all experiments"""
    try:
        successful_results = [r for r in all_results if not r.get("failed", False)]
        
        with open(results_file, 'a') as f:
            f.write("\n" + "="*80 + "\n")
            f.write("SUMMARY\n")
            f.write("="*80 + "\n")
            f.write(f"Total experiments: {len(all_results)}\n")
            f.write(f"Successful: {len(successful_results)}\n")
            f.write(f"Failed: {len(all_results) - len(successful_results)}\n\n")
            
            if successful_results:
                # Sort by performance (handle invalid values)
                def get_fde_score(result):
                    metrics = result.get("val_metrics", {})
                    fde = metrics.get("val_minFDE6", float('inf'))
                    if isinstance(fde, (int, float)) and not (np.isnan(fde) or np.isinf(fde)):
                        return fde
                    return float('inf')
                
                successful_results.sort(key=get_fde_score)
                
                f.write("TOP 5 CONFIGURATIONS:\n")
                f.write("-" * 50 + "\n")
                
                for i, result in enumerate(successful_results[:5]):
                    metrics = result.get("val_metrics", {})
                    config = result["config"]
                    
                    f.write(f"\nRank {i+1} (Experiment {result['experiment_id']}):\n")
                    
                    # Safe metric writing
                    for metric_name in ["val_minFDE6", "val_MR"]:
                        metric_value = metrics.get(metric_name, "N/A")
                        if isinstance(metric_value, (int, float)):
                            if np.isnan(metric_value) or np.isinf(metric_value):
                                f.write(f"  {metric_name}: invalid\n")
                            else:
                                f.write(f"  {metric_name}: {metric_value:.4f}\n")
                        else:
                            f.write(f"  {metric_name}: {metric_value}\n")
                    
                    f.write(f"  Parameters: {result.get('model_params', 'N/A'):,}\n")
                    f.write(f"  Config: {config}\n")
            
            f.write(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    except Exception as e:
        print(f"Error writing summary: {e}")


def analyze_results(results):
    """Analyze and print the best configurations"""
    try:
        successful_results = [r for r in results if not r.get("failed", False)]
        
        if not successful_results:
            print("No successful experiments!")
            return
        
        # Safe sorting
        def get_fde_score(result):
            metrics = result.get("val_metrics", {})
            fde = metrics.get("val_minFDE6", float('inf'))
            if isinstance(fde, (int, float)) and not (np.isnan(fde) or np.isinf(fde)):
                return fde
            return float('inf')
        
        successful_results.sort(key=get_fde_score)
        
        print(f"\n{'='*80}")
        print("RANDOM SEARCH RESULTS")
        print(f"{'='*80}")
        print(f"Total experiments: {len(results)}")
        print(f"Successful experiments: {len(successful_results)}")
        print(f"Failed experiments: {len(results) - len(successful_results)}")
        
        # Print top 5 configurations
        print(f"\nTOP 5 CONFIGURATIONS:")
        print("-" * 80)
        
        for i, result in enumerate(successful_results[:5]):
            metrics = result.get("val_metrics", {})
            config = result["config"]
            
            print(f"\nRank {i+1}:")
            
            # Safe metric printing
            for metric_name in ["val_minFDE6", "val_MR"]:
                metric_value = metrics.get(metric_name, "N/A")
                if isinstance(metric_value, (int, float)):
                    if np.isnan(metric_value) or np.isinf(metric_value):
                        print(f"  {metric_name}: invalid")
                    else:
                        print(f"  {metric_name}: {metric_value:.4f}")
                else:
                    print(f"  {metric_name}: {metric_value}")
            
            print(f"  Parameters: {result.get('model_params', 'N/A'):,}")
            print(f"  Config: {config}")
    
    except Exception as e:
        print(f"Error analyzing results: {e}")


if __name__ == "__main__":
    main()