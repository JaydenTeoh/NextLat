"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

Example usage:
python train.py --config path_to_your_config.yaml your.overrides=here

Or run with Lightning Fabric:
fabric run --strategy ddp --devices 4 --precision bf16-mixed train.py --config path_to_your_config.yaml your.overrides=here
"""

import os
import time
import torch
import wandb
import yaml
import wandb
import lightning as L
import itertools
from argparse import ArgumentParser
from omegaconf import OmegaConf, DictConfig
from lightning.fabric.strategies.model_parallel import ModelParallelStrategy
from lightning.fabric.loggers import CSVLogger
from wandb.integration.lightning.fabric import WandbLogger

from core_train import Trainer, initialize_model, shard_model
from train_probe import ProbeTrainer, initialize_probe_model
from data.tinystories import TinyStoriesDataModule
from data.stargraph import StarGraphDataModule
from data.fineweb import FineWebDataModule
from data.countdown import CountdownDataModule
from data.manhattan_dataset import ManhattanDataModule
from data.a5_data import A5DataModule

DATAMODULES = {
    "tinystories": TinyStoriesDataModule,
    "stargraph": StarGraphDataModule,
    "fineweb10B": FineWebDataModule,
    "fineweb100B": FineWebDataModule,
    "finewebedu": FineWebDataModule,
    "countdown": CountdownDataModule,
    "manhattan": ManhattanDataModule,
    "a5": A5DataModule,
}


def flatten(config):
    """Flatten a nested dictionary."""
    flat_config = {}
    for k, v in config.items():
        if isinstance(v, dict) or OmegaConf.is_dict(v):
            for k2, v2 in flatten(v).items():
                flat_config[f"{k}.{k2}"] = v2
        else:
            flat_config[k] = v
    return flat_config


def grid_to_list(config):
    """
    Convert a config with grid search parameters to a list of configs.
    Returns (config_list, sweep_param_names) where sweep_param_names are the parameters that were swept.
    """
    flat_config = flatten(config)
    iter_overwrites = {}
    flat_overwrites = {}

    for k, v in flat_config.items():
        if isinstance(v, list) or OmegaConf.is_list(v):
            iter_overwrites[k] = v
        else:
            flat_overwrites[k] = v

    if not iter_overwrites:
        return [config], []

    product_values = list(itertools.product(*iter_overwrites.values()))
    config_list = []
    sweep_param_names = list(iter_overwrites.keys())

    for values in product_values:
        overwrite_dict = dict(zip(iter_overwrites.keys(), values))
        overwrite_dict.update(flat_overwrites)
        config_list.append(overwrite_dict)

    return config_list, sweep_param_names


def do_train(
    config: DictConfig,
    hide_progress_bar: bool = False,
    use_sharding: bool = False,
    checkpoint_path: str = None,
    sweep_name: str = None,
):
    # Initialize logging
    experiment_name = config.trainer.experiment_name if not None else "run"
    if sweep_name:
        experiment_name = experiment_name + f"-{sweep_name}"
    if "seed" not in experiment_name:
        experiment_name = experiment_name + f"-seed{config.seed}"
    # experiment_name = experiment_name + str(time.time())
    loggers = []
    if config.trainer.log_to_file:
        loggers.append(
            CSVLogger(
                root_dir=config.trainer.out_dir,
                name=experiment_name,
                flush_logs_every_n_steps=1,
            )
        )
    if config.trainer.log_to_wandb:
        loggers.append(
            WandbLogger(
                project=config.trainer.wandb_project,
                name=experiment_name,
                group=(
                    config.trainer.experiment_name
                    if config.trainer.experiment_name is not None
                    else "unnamed"
                ),
                tags=config.trainer.wandb_tags if not None else [],
                config=OmegaConf.to_container(config),
            )
        )

    config.trainer.experiment_name = experiment_name

    # Initialize Fabric
    if use_sharding:
        # Sharding must be defined before creating Fabric
        strategy = ModelParallelStrategy(
            parallelize_fn=shard_model, save_distributed_checkpoint=False
        )
        fabric = L.Fabric(loggers=loggers, strategy=strategy)
        fabric.print("Training with sharded model")
    else:
        # GPUs, parallelism, and precision are set by the "fabric run" command
        fabric = L.Fabric(loggers=loggers)

    # Calculate per device batch size
    assert (
        config.data.effective_batch_size % fabric.world_size == 0
    ), f"effective_batch_size {config.data.effective_batch_size} must be divisible by DDP world size {fabric.world_size}"
    config.data.device_batch_size = (
        config.data.effective_batch_size // fabric.world_size
    )

    # Calculate micro batch size
    assert (
        config.data.device_batch_size % config.data.gradient_accum_steps == 0
    ), f"device_batch_size {config.data.device_batch_size} must be divisible by gradient_accum_steps {config.data.gradient_accum_steps}"
    config.data.micro_batch_size = (
        config.data.device_batch_size // config.data.gradient_accum_steps
    )

    # Print config
    fabric.print(yaml.dump(OmegaConf.to_container(config)))

    tokens_per_update = config.data.effective_batch_size * config.model.block_size
    fabric.print(f"World size: {fabric.world_size}")
    fabric.print(f"Effective batch size: {config.data.effective_batch_size}")
    fabric.print(f"Per-GPU device batch size: {config.data.device_batch_size}")
    fabric.print(
        f"Batch size per gradient accumulation step: {config.data.micro_batch_size}"
    )
    fabric.print(f"Batch size to accumulate pairs: {config.data.pair_batch_size}")
    fabric.print(f"Tokens per gradient update: {tokens_per_update}")

    # Initialize PyTorch settings
    seed_offset = fabric.global_rank
    fabric.seed_everything(int(config.seed) + int(seed_offset))
    torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
    torch._dynamo.config.cache_size_limit = 16  # allow more recompiles

    # Load dataset
    assert (
        config.data.dataset in DATAMODULES
    ), f"Dataset {config.data.dataset} not supported"
    DataModuleClass = DATAMODULES[config.data.dataset]
    datamodule = DataModuleClass(fabric, config)
    datamodule.update_config(config)
    tokenizer = datamodule.get_tokenizer()

    # Create output directory
    experiment_out_dir = os.path.join(config.trainer.out_dir, experiment_name)
    if fabric.global_rank == 0:
        os.makedirs(experiment_out_dir, exist_ok=True)
    fabric.print(f"Output directory: {experiment_out_dir}")

    # dump config to out_dir
    if fabric.global_rank == 0:
        OmegaConf.save(
            config, os.path.join(experiment_out_dir, "materialized_config.yaml")
        )

    # wandb login
    if (
        config.trainer.log_to_wandb
        and fabric.global_rank == 0
        and "WANDB_API_KEY" in os.environ
    ):
        wandb.login(
            key=os.environ.get("WANDB_API_KEY"),
            host=os.environ.get("WANDB_HOST"),
            timeout=0,
        )

    # Create model and trainer
    if config.trainer.train_probe:
        # Load model from checkpoint
        assert os.path.isfile(
            checkpoint_path
        ), f"Checkpoint file {checkpoint_path} not found"
        fabric.print(
            f"Training probe. Loading frozen model from checkpoint {checkpoint_path}"
        )
        model = initialize_model(
            fabric,
            config,
            tokenizer,
            initialize_optimizer=False,
            checkpoint_path=checkpoint_path,
        )
    else:
        fabric.barrier()
        model = initialize_model(
            fabric,
            config,
            tokenizer,
            initialize_optimizer=True,
            checkpoint_path=checkpoint_path,
        )

    if config.trainer.train_probe:
        probe = initialize_probe_model(fabric, config, model, tokenizer)
        trainer = ProbeTrainer(
            fabric=fabric,
            config=config,
            model=probe,
            show_progress_bar=(fabric.local_rank == 0 and not hide_progress_bar),
        )
    else:
        trainer = Trainer(
            fabric=fabric,
            config=config,
            model=model,
            show_progress_bar=(fabric.local_rank == 0 and not hide_progress_bar),
        )

    # Start training
    trainer.train(datamodule)


if __name__ == "__main__":
    parser = ArgumentParser(
        prog="BST_Trainer",
        description="Belief State Transformer (BST) trainer",
    )
    parser.add_argument("-c", "--config", required=True, help="Path to config file")
    parser.add_argument("--no_pbar", action="store_true", help="Disable progress bar")
    parser.add_argument("--shard", action="store_true", help="Shard the model")
    parser.add_argument(
        "--checkpoint_path", required=False, help="Path to checkpoint file"
    )
    args, conf_cli = parser.parse_known_args()

    # Load base config
    base_config = OmegaConf.load(args.config)

    # Check if config has sweep section
    has_sweep = "sweep" in base_config and base_config.sweep is not None

    if has_sweep:
        # Handle sweep mode - collect all combinations first, then run sequentially
        list_of_sweeps = base_config.pop("sweep", [])
        all_configs = []

        # Collect all sweep configurations
        all_configs = []
        all_sweep_param_names = set()  # Track all sweep parameter names

        for sweep in list_of_sweeps:
            sweep_configs, sweep_param_names = grid_to_list(sweep)
            all_configs.extend(sweep_configs)
            all_sweep_param_names.update(sweep_param_names)

        print(f"Collected {len(all_configs)} configurations for hyperparameter sweep")

        # Run each configuration sequentially
        for i, sweep_config_dict in enumerate(all_configs):
            print(f"\n{'='*60}")
            print(f"Running configuration {i+1}/{len(all_configs)}")
            print(f"{'='*60}")

            # Merge with defaults and CLI overrides
            default = OmegaConf.load("defaults.yaml")
            cli = OmegaConf.from_dotlist(conf_cli)
            print("sweep_config_dict", sweep_config_dict)

            # Convert flat dict back to nested structure
            nested_config = {}
            for key, value in sweep_config_dict.items():
                if "." in key:
                    parts = key.split(".")
                    current = nested_config
                    for part in parts[:-1]:
                        if part not in current:
                            current[part] = {}
                        current = current[part]
                    current[parts[-1]] = value
                else:
                    nested_config[key] = value

            sweep_config = OmegaConf.create(nested_config)
            config = OmegaConf.merge(default, base_config, sweep_config, cli)
            print("config", config)

            # Create descriptive experiment name with hyperparameter values
            if config.trainer.experiment_name:
                # Get values for sweep parameters
                sweep_params = []
                for param_name in all_sweep_param_names:
                    param_value = OmegaConf.select(config, param_name)
                    if param_value is not None:
                        # Extract just the parameter name (not the full path)
                        param_key = param_name.split(".")[-1]
                        sweep_params.append(f"{param_key}{param_value}")

                # Create descriptive name
                sweep_name = "_".join(sweep_params) if sweep_params else None

            # Run training for this configuration
            do_train(
                config,
                hide_progress_bar=args.no_pbar,
                use_sharding=args.shard,
                checkpoint_path=args.checkpoint_path,
                sweep_name=sweep_name,
            )

            print(f"Completed configuration {i+1}/{len(all_configs)}")

        print(f"\n{'='*60}")
        print(f"Hyperparameter sweep completed! Ran {len(all_configs)} configurations.")
        print(f"{'='*60}")
    else:
        # Normal training mode
        default = OmegaConf.load("defaults.yaml")
        cli = OmegaConf.from_dotlist(conf_cli)
        config = OmegaConf.merge(default, base_config, cli)
        do_train(
            config,
            hide_progress_bar=args.no_pbar,
            use_sharding=args.shard,
            checkpoint_path=args.checkpoint_path,
        )
