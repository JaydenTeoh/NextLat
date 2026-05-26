"""contains core training/inference logic used for multiple datasets/tasks"""

import gc
import inspect
import math
import os
import time
import torch
import wandb
import lightning as L
from datetime import datetime
from lightning.fabric.strategies.model_parallel import ModelParallelStrategy
from torch.optim.lr_scheduler import LambdaLR
from torch.distributed.fsdp import fully_shard
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Any, Callable, Dict, Optional, Tuple
from collections import defaultdict

from models.model_base import ModelBase
from models.model_bst import BST, BSTConfig
from models.model_gpt import GPT, GPTConfig, Block
from models.model_nextlat import NextLat, NextLatConfig
from models.model_mtp_gloeckle import MTPGloeckle, MTPGloeckleConfig
from models.model_mtp_jtp import JTP, MTPJTPConfig
from models.model_speculative import SpeculativeModel


def initialize_model(
    fabric: L.Fabric,
    config,
    tokenizer,
    initialize_optimizer=True,
    checkpoint_path: Optional[str] = None,
):
    fabric.print(f"Initializing model with config: {config}")

    if config.use_bst:
        ModelClass = BST
        ModelConfigClass = BSTConfig
        single_gap_modes = ["next_token", "eos"]
        assert (
            config.model.bst_single_gap_prediction_mode in single_gap_modes
        ), f"BST single gap mode must be one of {single_gap_modes}"
    elif config.use_nextlat:
        ModelClass = NextLat
        ModelConfigClass = NextLatConfig
    elif config.use_mtp_gloeckle:
        ModelClass = MTPGloeckle
        ModelConfigClass = MTPGloeckleConfig
    elif config.use_mtp_jtp:
        ModelClass = JTP
        ModelConfigClass = MTPJTPConfig
    else:
        ModelClass = GPT
        ModelConfigClass = GPTConfig
        gpt_modes = ["next_token", "fim"]
        assert (
            config.model.gpt_mode in gpt_modes
        ), f"GPT mode must be one of {gpt_modes}"

    # start with model_args from command line
    model_args = dict(
        n_layer=config.model.n_layer,
        n_head=config.model.n_head,
        n_embd=config.model.n_embd,
        block_size=config.model.block_size,
        bias=config.model.bias,
        vocab_size=config.model.vocab_size,
        dropout=config.model.dropout,
        eos_token_id=tokenizer.eos_token_id,
        use_fused=config.trainer.use_fused_kernels,
    )

    if config.use_bst:
        # BST-specific parameters
        model_args = {
            **model_args,
            "context_length": config.model.context_length,
            "bst_pair_minimum_gap": config.model.bst_pair_minimum_gap,
            "bst_pair_maximum_gap": config.model.bst_pair_maximum_gap,
            "bst_pair_subsample_rate": config.model.bst_pair_subsample_rate,
            "bst_single_gap_prediction_mode": config.model.bst_single_gap_prediction_mode,
        }
    elif config.use_nextlat:
        # NextLat-specific parameters
        model_args = {
            **model_args,
            "context_length": config.model.context_length,
            "mtp_horizon": config.model.mtp_horizon,
            "lambda_kl": config.model.lambda_kl,
            "lambda_mse": config.model.lambda_mse,
            "lambda_ce": config.model.lambda_ce,
            "proj_factor": config.model.proj_factor,
            "compute_hidden_state_rank": config.model.compute_hidden_state_rank,
        }
    elif config.use_mtp_gloeckle:
        model_args = {
            **model_args,
            "context_length": config.model.context_length,
            "mtp_horizon": config.model.mtp_horizon,
            "mtp_lambda": config.model.mtp_lambda,
            "compute_hidden_state_rank": config.model.compute_hidden_state_rank,
        }
    elif config.use_mtp_jtp:
        model_args = {
            **model_args,
            "context_length": config.model.context_length,
            "mtp_horizon": config.model.mtp_horizon,
            "mtp_lambda": config.model.mtp_lambda,
            "compute_hidden_state_rank": config.model.compute_hidden_state_rank,
        }
    else:
        # GPT-specific parameters
        model_args = {
            **model_args,
            "context_length": config.model.context_length,
            "goal_range": config.data.goal_range,
            "fim_token_id": (
                -1
                if config.model.gpt_mode != "fim"
                else tokenizer.convert_tokens_to_ids("<|fim|>")
            ),
            "is_fim_mode": config.model.gpt_mode == "fim",
            "compute_hidden_state_rank": config.model.compute_hidden_state_rank,
        }
    model_config = ModelConfigClass(**model_args)

    # Load a checkpoint file
    if checkpoint_path:
        assert os.path.isfile(
            checkpoint_path
        ), f"Checkpoint file {checkpoint_path} does not exist"

        # Create an empty model because we will load the weights from checkpoint
        with fabric.init_module(empty_init=True):
            model = ModelClass(model_config)

    # Resume from a previous training run
    elif config.trainer.init_from == "resume":
        recovery_ckpt_pointer = os.path.join(config.trainer.out_dir, "recovery_ckpt")
        latest_ckpt_pointer = os.path.join(config.trainer.out_dir, "latest_ckpt")

        # Use the recovery checkpoint if it exists
        if os.path.isfile(recovery_ckpt_pointer):
            with open(recovery_ckpt_pointer, "r") as f:
                checkpoint_path = f.read().strip()
            assert os.path.isfile(
                checkpoint_path
            ), f"Checkpoint file {checkpoint_path} does not exist"
            fabric.print(f"Resuming from recovery checkpoint {checkpoint_path}")

        # Otherwise, use the latest validation checkpoint if it exists
        elif os.path.isfile(latest_ckpt_pointer):
            with open(latest_ckpt_pointer, "r") as f:
                checkpoint_path = f.read().strip()
            assert os.path.isfile(
                checkpoint_path
            ), f"Checkpoint file {checkpoint_path} does not exist"
            fabric.print(
                f"Resuming from previous validation checkpoint {checkpoint_path}"
            )

        # If no checkpoint file is found, initialize a new model
        else:
            fabric.print(f"Could not find checkpoint file {recovery_ckpt_pointer}")
            fabric.print(f"Could not find checkpoint file {latest_ckpt_pointer}")
            checkpoint_path = None

        # Empty init if we have found a checkpoint
        with fabric.init_module(empty_init=(checkpoint_path is not None)):
            model = ModelClass(model_config)

    # Initialize a new model from scratch
    elif config.trainer.init_from == "scratch":
        with fabric.init_module():
            model = ModelClass(model_config)

    # Initialize from OpenAI GPT-2 weights
    elif config.trainer.init_from.startswith("gpt2"):  # currently broken @dayan
        fabric.print(f"Initializing from OpenAI GPT-2 weights: {config.init_from}")
        assert not config.use_bst, "BST not supported with GPT-2 weights"
        override_args = dict(dropout=config.model.dropout)
        model = ModelClass.from_pretrained(config.trainer.init_from, override_args)
        # read off the created config params, so we can store them into checkpoint correctly
        for k in ["n_layer", "n_head", "n_embd", "block_size", "bias", "vocab_size"]:
            model_args[k] = getattr(model.config, k)

    else:
        raise ValueError(
            f"Invalid init_from value: {config.trainer.init_from}. Must be 'resume', 'scratch', or 'gpt2'."
        )

    # crop down the model block size if desired, using model surgery
    # so that the checkpoint will have the right value
    if config.model.block_size < model.config.block_size:
        assert not config.use_bst, "cropping block size not supported for BST"
        model.crop_block_size(config.model.block_size)
        model_args["block_size"] = config.model.block_size

    fabric.print(
        f"Number of parameters (including embedding): {model.get_num_params(non_embedding=False):,}"
    )
    fabric.print(
        f"Number of parameters (excluding embedding): {model.get_num_params(non_embedding=True):,}"
    )

    # Compile must occur before fabric.setup()
    if config.trainer.compile:
        fabric.print("Compiling model")
        model.compile()

    # FSDP requires model to be setup before initializing the optimizer
    if isinstance(fabric.strategy, ModelParallelStrategy):
        model.setup_fabric(fabric)

    # Initialize optimizer
    if initialize_optimizer:
        is_device_cuda = fabric.device.type == "cuda"
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        # This is native PyTorch fused optimizer, not use_fused_kernels in config
        use_fused = fused_available and is_device_cuda

        model.configure_optimizers(
            weight_decay=config.optimizer.weight_decay,
            learning_rate=config.optimizer.learning_rate,
            betas=(config.optimizer.beta1, config.optimizer.beta2),
            use_fused=use_fused,
            optimizer_type=config.optimizer.optimizer_type,
            mu=config.optimizer.mu,
            nesterov=config.optimizer.nesterov,
            adjust_lr_fn=config.optimizer.adjust_lr_fn,
        )

    # Setup model and optimizer
    # In the FSDP case, model is already setup and this sets up only the optimizer
    model.setup_fabric(fabric)

    # Load checkpoint file
    if checkpoint_path:
        model.load_checkpoint(checkpoint_path)
        fabric.print(f"Loaded checkpoint from {checkpoint_path}")
    else:
        fabric.print("Initialized a new model from scratch")

    return model


def shard_model(
    module: torch.nn.Module, device_mesh: torch.distributed.device_mesh.DeviceMesh
):
    """
    Function to define the sharding strategy for the model.

    Given a 2D device mesh of (nodes, gpus per node), fully_shard will:
        - Replicate across nodes (data parallel)
        - Shard across GPUs within a node (FSDP)
    """

    # Function to shard individual transformer layers recursively
    # This lets us only gather full weights one layer at a time
    def _shard_recursive(module: torch.nn.Module):
        for submodule in module.children():
            if isinstance(submodule, Block):
                submodule = fully_shard(
                    submodule,
                    mesh=device_mesh,
                    reshard_after_forward=True,
                )
            else:
                _shard_recursive(submodule)

    # Shard the submodules
    _shard_recursive(module)

    # Shard the top level module
    fully_shard(module, mesh=device_mesh, reshard_after_forward=False)

    return module


class Trainer:
    """Trainer class for training/validating BST and GPT models"""

    def __init__(
        self,
        fabric: L.Fabric,
        config,
        model: ModelBase,
        show_progress_bar: bool = True,
    ):
        # Check that model has optimizer
        assert (
            hasattr(model, "optimizer") and model.optimizer is not None
        ), "Model must be initialized with optimizer"

        # Initialization state
        self.fabric = fabric
        self.config = config
        self.model = model
        self.show_progress_bar = show_progress_bar

        # Training loop state
        self.train_dataloader: DataLoader = None
        self.val_dataloader: DataLoader = None
        self.tokenizer = None
        self.prepare_batch_func: Optional[Callable] = None
        self.epoch: int = 0
        self.step: int = self.model.training_steps
        self.last_losses: Dict[str, float] = {}
        self.best_val_loss: Optional[float] = None
        optimizers = (
            list(self.model.optimizer)
            if isinstance(self.model.optimizer, (list, tuple))
            else [self.model.optimizer]
        )
        lr_lambda = self._select_lr_lambda()
        self.lr_schedulers = [LambdaLR(opt, lr_lambda=lr_lambda) for opt in optimizers]
        scheduler_state = self.model.lr_scheduler_state
        if scheduler_state is not None:
            if isinstance(scheduler_state, list):
                for scheduler, state in zip(self.lr_schedulers, scheduler_state):
                    scheduler.load_state_dict(state)
            elif len(self.lr_schedulers) == 1:
                self.lr_schedulers[0].load_state_dict(scheduler_state)
        # Align optimizer group lrs to current training step without deprecated epoch arg.
        for scheduler in self.lr_schedulers:
            while scheduler.last_epoch < self.step - 1:
                scheduler.step()

        # Checkpointing state
        self.latest_checkpoint_path = None
        self.best_checkpoint_path = None
        self.recovery_checkpoint_path = None
        self.checkpoints_to_always_keep = set()

    def train(self, datamodule):
        """
        Call this to start training.
        """
        # Initialize dataloaders
        self.train_dataloader = datamodule.train_dataloader()
        self.val_dataloader = datamodule.val_dataloader()
        if (
            self.config.data.dataset == "a5"
            or self.config.data.dataset == "stargraph"
            or self.config.data.dataset == "countdown"
            or self.config.data.dataset == "manhattan"
        ) and self.config.data.test_generalization:
            self.generalization_dataloader = datamodule.generalization_dataloader()
        self.tokenizer = datamodule.get_tokenizer()

        # Check if datamodule has a prepare_batch function
        if hasattr(datamodule, "prepare_batch"):
            self.prepare_batch_func = datamodule.prepare_batch
            self.fabric.print(
                f"Using prepare batch function {type(datamodule).__name__}.prepare_batch()"
            )
        else:
            self.prepare_batch_func = None

        # Do training or validation
        if self.config.trainer.val_only:
            self.fabric.print("Running validation only, not training model")
            validation_logs = self._validation_loop()
            self.fabric.print(validation_logs)
        else:
            self.fabric.print(f"Starting training from step={self.step}")
            self._train_loop()

            self.fabric.print("Training complete, running final validation")
            val_logs = self._validation_loop()

            # Save final model checkpoint as wandb artifact
            if self.config.trainer.log_to_wandb and self.fabric.global_rank == 0:
                # Initialize wandb artifact
                artifact = wandb.Artifact(
                    name=self.config.trainer.experiment_name,
                    type="model",
                    metadata={
                        "config": self.config,
                        "validation_logs": val_logs,
                        "best_val_loss": self.best_val_loss,
                        **self.last_losses,
                    },
                )
                # Get the latest checkpoint filename
                latest_ckpt_pointer = os.path.join(
                    self.config.trainer.out_dir, "latest_ckpt"
                )
                if os.path.isfile(latest_ckpt_pointer):
                    # Upload the checkpoint
                    with open(latest_ckpt_pointer, "r") as f:
                        checkpoint_path = f.read().strip()
                    assert os.path.isfile(
                        checkpoint_path
                    ), f"Checkpoint file {checkpoint_path} does not exist"
                    self.fabric.print(f"Uploading {checkpoint_path} to wandb artifact")
                    artifact.add_file(
                        checkpoint_path,
                        name=os.path.basename(checkpoint_path),
                    )
                    wandb.run.log_artifact(artifact)
                else:
                    # Checkpoint file not found
                    self.fabric.print(
                        f"Warning: {latest_ckpt_pointer} file not found, not saving wandb artifact"
                    )
                wandb.finish()

    def _train_loop(self):
        """
        Main training loop.
        Iterate over the train dataloader for the configured number of batches.

        For each batch:
            - If we have reached the validation interval, run validation
            - Split batch into micro batches and accumulate gradients
            - Run optimizer step
            - Do logging
        """
        # Set model to train mode
        self.model.train()

        # Sum of train loss over log_interval batches
        running_loss_sums = defaultdict(lambda: torch.zeros(1).to(self.fabric.device))
        speed_window_train_time = 0.0
        speed_window_train_steps = 0
        # Initialize progress bar
        if self.show_progress_bar:
            pbar = tqdm(total=self.config.trainer.val_interval, leave=False)

        # If resuming from a previous checkpoint, fast-forward data
        if self.step > 0:
            do_fast_forward = True
            fast_forward_steps = 0
        else:
            do_fast_forward = False

        while True:
            # Do one epoch over the entire training dataloader
            for batch in self.train_dataloader:
                # Fast forward data
                if do_fast_forward:
                    # Increment fast forward counter until we reach the current step
                    fast_forward_steps += 1
                    if fast_forward_steps >= self.step:
                        self.fabric.print(
                            f"Fast forwarded data to step {fast_forward_steps}"
                        )
                        do_fast_forward = False
                    # Skip this batch
                    continue

                # Garbage collect to free up memory
                if (
                    self.config.trainer.garbage_collect > 0
                    and self.step % self.config.trainer.garbage_collect == 0
                ):
                    gc.collect()

                # Validation
                validation_logs = {}
                if (
                    self.step % self.config.trainer.val_interval == 0 and self.step > 0
                ) or self.config.trainer.val_only:
                    # Close the previous training progress bar
                    if self.show_progress_bar:
                        pbar.close()

                    validation_logs = self._validation_loop()
                    self.model.train()

                    if self.config.trainer.garbage_collect > 0:
                        gc.collect()

                    # Initialize a new progress bar
                    if self.show_progress_bar:
                        pbar = tqdm(total=self.config.trainer.val_interval, leave=False)

                # Prepare batch
                iter_train_start = time.perf_counter()
                if self.prepare_batch_func is not None:
                    batch = self.prepare_batch_func(batch)

                # Accumulate gradients over gradient_accum_steps
                for accum_step in range(self.config.data.gradient_accum_steps):
                    start_idx = accum_step * self.config.data.micro_batch_size
                    end_idx = (accum_step + 1) * self.config.data.micro_batch_size
                    sub_batch = batch[start_idx:end_idx]

                    # Only sync gradients for the last step
                    no_sync = accum_step < self.config.data.gradient_accum_steps - 1
                    if self.config.data.dataset == "a5":
                        # Dataset yields concat(input, target); model consumes input, predicts target.
                        split = sub_batch.shape[-1] // 2
                        targets = sub_batch[:, split:]
                        sub_batch = sub_batch[:, :split]
                    else:
                        targets = sub_batch[:, 1:]
                    losses_dict = self.model.compute_loss(
                        sub_batch,
                        targets=targets,
                        pair_batch_size=self.config.data.pair_batch_size,
                        backpropagate=True,
                        loss_div=self.config.data.gradient_accum_steps,
                        no_sync=no_sync,
                    )
                    for k, v in losses_dict.items():
                        running_loss_sums[k] += v

                # Optimizer step
                # This should increment model.training_steps
                grad_clip = self.config.optimizer.grad_clip
                grad_norm = self.model.optimizer_step(grad_clip)
                speed_window_train_time += time.perf_counter() - iter_train_start
                speed_window_train_steps += 1
                first_optimizer = (
                    self.model.optimizer[0]
                    if isinstance(self.model.optimizer, (list, tuple))
                    else self.model.optimizer
                )
                lr = first_optimizer.param_groups[0]["lr"]
                for scheduler in self.lr_schedulers:
                    scheduler.step()
                # Logging
                if self.step % self.config.trainer.log_interval == 0 and self.step > 0:
                    # Train loss has been accumulated over log_interval batches
                    for k, v in running_loss_sums.items():
                        self.last_losses[k] = (
                            v.item() / self.config.trainer.log_interval
                        )
                        running_loss_sums[k].zero_()

                    custom_metrics = self.model.get_custom_metrics()
                    elapsed_time = max(speed_window_train_time, 1e-8)
                    steps_in_window = max(speed_window_train_steps, 1)
                    steps_per_sec = steps_in_window / elapsed_time
                    log_dict = {
                        "step": self.step,
                        "epoch": self.epoch,
                        "lr": lr,
                        "steps_per_sec": steps_per_sec,
                        **self.last_losses,
                        **validation_logs,
                        **custom_metrics,
                    }
                    if self.config.data.dataset.startswith("fineweb"):
                        tokens_per_step = (
                            self.config.data.effective_batch_size
                            * self.config.model.block_size
                        )
                        log_dict["tokens_per_sec"] = (
                            steps_in_window * tokens_per_step
                        ) / elapsed_time
                        log_dict["tokens_seen"] = (self.step + 1) * tokens_per_step
                    if grad_norm is not None:
                        log_dict["grad_norm"] = grad_norm

                    log_dict.update(self._next_token_perplexity_metrics(log_dict))
                    self.fabric.log_dict(log_dict)
                    speed_window_train_time = 0.0
                    speed_window_train_steps = 0

                # Update progress bar
                progress_str = f"{datetime.now()} Step: {self.step} Epoch: {self.epoch} Train loss: {self.last_losses.get('loss', 0.0):.4f}"
                if self.show_progress_bar:
                    pbar.set_description(progress_str)
                    pbar.update(1)
                else:
                    self.fabric.print(progress_str)

                # Increment batch count
                self.step += 1
                assert (
                    self.step == self.model.training_steps
                ), "Bug in training loop: trainer step count does not match model step count"
                if self.step > self.config.trainer.train_batches:
                    # End of training
                    return

                # Save recovery checkpoint
                if (
                    self.config.trainer.save_recovery_checkpoint > 0
                    and self.step % self.config.trainer.save_recovery_checkpoint == 0
                ):
                    self._save_recovery_checkpoint()

            # End of one full iteration over train dataloader
            self.epoch += 1

    @torch.inference_mode()
    def _validation_loop(self) -> Dict[str, Any]:
        """
        Main validation loop.
        This is called every val_interval steps during training.

        For validation, we do the following:
            - Compute validation loss
            - For BST, compute next/prev token prediction loss
            - Save checkpoint
        """
        # Set model to eval mode
        self.model.eval()

        if self.config.trainer.val_batches > 0:
            # Divide total validation batches by world size
            n_val_batches = self.config.trainer.val_batches // self.fabric.world_size
        else:
            # If validation batches are not specified, validate over entire dataloader
            n_val_batches = None

        validation_logs = {}
        batch_count = 0
        total_losses = defaultdict(lambda: torch.zeros(1).to(self.fabric.device))
        spec_eval_num_samples = int(self.config.trainer.get("spec_eval_num_samples", 0))
        spec_eval_enabled = bool(
            self.config.trainer.get("spec_eval_enabled", False)
        ) and self.config.data.dataset.startswith("fineweb")
        spec_eval_supported = isinstance(self.model, SpeculativeModel)
        disable_pbar = (
            True if not self.show_progress_bar else None
        )  # None means automatic

        # Main validation loop
        for batch in tqdm(
            self.val_dataloader,
            desc="Validation",
            total=n_val_batches or len(self.val_dataloader),
            leave=False,
            disable=disable_pbar,
        ):
            if self.prepare_batch_func is not None:
                batch = self.prepare_batch_func(batch)

            # Accumulate loss over gradient_accum_steps
            for accum_step in range(self.config.data.gradient_accum_steps):
                start_idx = accum_step * self.config.data.micro_batch_size
                end_idx = (accum_step + 1) * self.config.data.micro_batch_size
                sub_batch = batch[start_idx:end_idx]

                if self.config.data.dataset == "a5":
                    split = sub_batch.shape[-1] // 2
                    targets = sub_batch[:, split:]
                    sub_batch = sub_batch[:, :split]
                else:
                    targets = sub_batch[:, 1:]

                loss = self.model.compute_loss(
                    sub_batch,
                    targets=targets,
                    pair_batch_size=self.config.data.pair_batch_size,
                    backpropagate=False,
                    loss_div=self.config.data.gradient_accum_steps,
                )
                for k, v in loss.items():
                    total_losses["val/" + k] += v.to(self.fabric.device)

            # End loop if limit_batches is reached
            batch_count += 1
            if n_val_batches is not None and batch_count >= n_val_batches:
                break

        if self.config.trainer.val_printsamples:
            prefix = sub_batch[:, : self.config.model.context_length]

            self.model.evaluation_preds(
                sub_batch, self.config.model.context_length, sub_batch.shape[1] - 2
            )

        # Sync average loss across all GPUs by uniform mean of per-device averages
        # This assumes that each GPU has roughly the same number of batches
        device_avg_losses = {k: v / batch_count for k, v in total_losses.items()}
        global_avg_losses = {
            k: self.fabric.all_reduce(v, reduce_op="mean").item()
            for k, v in device_avg_losses.items()
        }
        validation_logs.update(global_avg_losses)
        val_loss = global_avg_losses["val/loss"]

        # For BST, compute next/prev token prediction loss separately with empty prefix/suffix
        if isinstance(self.model, BST):
            val_loss_next_token, val_loss_prev_token = self._bst_next_prev_token_loss()
            validation_logs["val/next_token_loss"] = val_loss_next_token
            validation_logs["val/prev_token_loss"] = val_loss_prev_token

        # Modular accuracy evaluation
        if self.step % self.config.trainer.test_interval == 0 and self.step > 0:
            # StarGraph
            if self.config.data.dataset == "stargraph":
                # Import and call evaluate_stargraph
                acc_logs = self.val_dataloader.dataset.evaluate_stargraph(
                    self.model,
                    self.val_dataloader,
                    self.config,
                    self.fabric,
                    show_progress_bar=self.show_progress_bar,
                )
                validation_logs.update(acc_logs)
                if self.config.data.test_generalization:
                    acc_logs = (
                        self.generalization_dataloader.dataset.evaluate_stargraph(
                            self.model,
                            self.generalization_dataloader,
                            self.config,
                            self.fabric,
                            show_progress_bar=self.show_progress_bar,
                            generalization=True,
                        )
                    )
                    validation_logs.update(acc_logs)
            # A5
            elif self.config.data.dataset == "a5":
                acc_logs = self.val_dataloader.dataset.evaluate_a5(
                    self.model,
                    self.val_dataloader,
                    self.config,
                    self.fabric,
                    show_progress_bar=self.show_progress_bar,
                )
                validation_logs.update(acc_logs)
                if self.config.data.test_generalization:
                    acc_logs = self.generalization_dataloader.dataset.evaluate_a5(
                        self.model,
                        self.generalization_dataloader,
                        self.config,
                        self.fabric,
                        show_progress_bar=self.show_progress_bar,
                        generalization=True,
                    )
                    validation_logs.update(acc_logs)
            # countdown
            elif self.config.data.dataset == "countdown":
                acc_logs = self.val_dataloader.dataset.evaluate_countdown(
                    self.model,
                    self.val_dataloader,
                    self.config,
                    self.fabric,
                    show_progress_bar=self.show_progress_bar,
                )
                validation_logs.update(acc_logs)
                if self.config.data.test_generalization:
                    acc_logs = (
                        self.generalization_dataloader.dataset.evaluate_countdown(
                            self.model,
                            self.generalization_dataloader,
                            self.config,
                            self.fabric,
                            show_progress_bar=self.show_progress_bar,
                            generalization=True,
                        )
                    )
                    validation_logs.update(acc_logs)
            # Manhattan
            elif self.config.data.dataset == "manhattan":
                acc_logs = self.generalization_dataloader.dataset.evaluate_manhattan(
                    self.model,
                    self.generalization_dataloader,
                    self.config,
                    self.fabric,
                    show_progress_bar=self.show_progress_bar,
                )
                validation_logs.update(acc_logs)
                if self.config.data.test_generalization:
                    acc_logs = self.generalization_dataloader.dataset.evaluate_manhattan_generalization(
                        self.model,
                        self.generalization_dataloader,
                        self.config,
                        self.fabric,
                        show_progress_bar=self.show_progress_bar,
                    )
                    validation_logs.update(acc_logs)

            # FineWeb speculative decoding benchmark for MTP-capable models.
            elif (
                self.config.data.dataset.startswith("fineweb")
                and spec_eval_enabled
                and spec_eval_supported
                and spec_eval_num_samples > 0
            ):
                spec_logs = self.val_dataloader.dataset.evaluate_speculative_fineweb(
                    self.model,
                    self.val_dataloader,
                    self.config,
                    self.fabric,
                    prepare_batch_func=self.prepare_batch_func,
                )
                validation_logs.update(spec_logs)

        validation_logs.update(self._next_token_perplexity_metrics(validation_logs))
        self.fabric.print(
            f"Step: {self.step} Epoch: {self.epoch} Train loss: {self.last_losses.get('loss', 0.0)} Validation loss: {val_loss}"
        )

        # Update best validation loss
        is_new_best = (
            True if self.best_val_loss is None else (val_loss < self.best_val_loss)
        )
        if is_new_best:
            self.best_val_loss = val_loss

        # Save checkpoint
        if not self.config.trainer.val_only and (
            self.config.trainer.save_last_checkpoint
            or (self.config.trainer.save_best_checkpoint and is_new_best)
            or (
                self.config.trainer.keep_checkpoint_steps
                and self.step in self.config.trainer.keep_checkpoint_steps
            )
        ):
            filename = f"ckpt_iter_{self.step}.pt"
            if self.config.trainer.save_best_checkpoint:
                filename = f"ckpt_iter_{self.step}_{val_loss:.4f}.pt"
            new_ckpt_path = self._save_checkpoint(filename)

            # Delete outdated checkpoints
            if not self.config.trainer.always_save_checkpoint:
                old_checkpoints = set()
                if (
                    self.config.trainer.save_last_checkpoint
                    and self.latest_checkpoint_path is not None
                    and (
                        self.latest_checkpoint_path != self.best_checkpoint_path
                        or is_new_best
                    )
                ):
                    old_checkpoints.add(self.latest_checkpoint_path)
                if (
                    self.config.trainer.save_best_checkpoint
                    and is_new_best
                    and self.best_checkpoint_path is not None
                ):
                    old_checkpoints.add(self.best_checkpoint_path)
                if self.fabric.global_rank == 0:
                    for ckpt in old_checkpoints:
                        if ckpt not in self.checkpoints_to_always_keep:
                            os.remove(ckpt)

            # Update latest and best checkpoint paths
            if self.config.trainer.save_last_checkpoint:
                self.latest_checkpoint_path = new_ckpt_path
            if self.config.trainer.save_best_checkpoint and is_new_best:
                self.best_checkpoint_path = new_ckpt_path
            if (
                self.config.trainer.keep_checkpoint_steps
                and self.step in self.config.trainer.keep_checkpoint_steps
            ):
                self.checkpoints_to_always_keep.add(new_ckpt_path)

        return validation_logs

    @staticmethod
    def _next_token_perplexity_metrics(metrics: Dict[str, Any]) -> Dict[str, float]:
        """
        Derive perplexity only from next-token losses.
        Examples:
          - next_token_loss -> ppl
          - val/next_token_loss -> val/ppl
        """
        out: Dict[str, float] = {}
        for key, value in metrics.items():
            if "next_token_loss" not in key:
                continue
            if not isinstance(value, (int, float)):
                continue
            ppl_key = key.replace("next_token_loss", "ppl")
            out[ppl_key] = float(math.exp(min(float(value), 20.0)))
        return out

    @torch.inference_mode()
    def _bst_next_prev_token_loss(self) -> Tuple[float, float]:
        """
        Compute next/previous token prediction loss for BST.
        The next token loss computed here is equivalent to the GPT validation loss.
        This lets us have a fair comparison to GPT.
        """
        assert isinstance(self.model, BST), "This function is only valid for BST"

        if self.config.trainer.val_batches > 0:
            # Divide total validation batches by world size
            n_val_batches = self.config.trainer.val_batches // self.fabric.world_size
        else:
            # If validation batches are not specified, validate over entire dataloader
            n_val_batches = None

        batch_count = 0
        total_next_loss = torch.zeros(1, device=self.fabric.device)
        total_prev_loss = torch.zeros(1, device=self.fabric.device)

        disable_pbar = (
            True if not self.show_progress_bar else None
        )  # None means automatic

        for batch in tqdm(
            self.val_dataloader,
            desc="BST next/prev token loss",
            total=n_val_batches or len(self.val_dataloader),
            leave=False,
            disable=disable_pbar,
        ):
            if self.prepare_batch_func is not None:
                batch = self.prepare_batch_func(batch)

            # Accumulate loss over gradient_accum_steps
            for accum_step in range(self.config.data.gradient_accum_steps):
                start_idx = accum_step * self.config.data.micro_batch_size
                end_idx = (accum_step + 1) * self.config.data.micro_batch_size
                sub_batch = batch[start_idx:end_idx]

                # Create tensor of -1 with length equal to sub_batch size
                fill_value = (
                    self.config.model.context_length
                    if self.config.model.context_length > 0
                    else -1
                )
                prefix_end_index = torch.full(
                    (sub_batch.size(0),),
                    fill_value=fill_value,
                    device=sub_batch.device,
                    dtype=torch.long,
                )
                neg_one = torch.full(
                    (sub_batch.size(0),),
                    fill_value=-1,
                    device=sub_batch.device,
                    dtype=torch.long,
                )

                # Compute next/prev token prediction loss
                # We do not have prefix/suffix as prompt, so set indices to -1
                next_token_losses, prev_token_losses = self.model.evaluation_loss(
                    sub_batch,
                    prefix_end_index=prefix_end_index,
                    suffix_start_index=neg_one,
                )

                # Calculate mean loss over sequences in sub-batch
                # Divide loss by gradient_accum_steps so we sum to the batch mean
                next_token_losses = (
                    next_token_losses.mean() / self.config.data.gradient_accum_steps
                )
                prev_token_losses = (
                    prev_token_losses.mean() / self.config.data.gradient_accum_steps
                )
                total_next_loss += next_token_losses
                total_prev_loss += prev_token_losses

            # End loop if limit_batches is reached
            batch_count += 1
            if n_val_batches is not None and batch_count >= n_val_batches:
                break

        # Sync average loss across all GPUs
        device_avg_next_loss = total_next_loss / batch_count
        device_avg_prev_loss = total_prev_loss / batch_count
        global_avg_next_loss, global_avg_prev_loss = self.fabric.all_reduce(
            (device_avg_next_loss, device_avg_prev_loss), reduce_op="mean"
        )

        return global_avg_next_loss.item(), global_avg_prev_loss.item()

    def _save_checkpoint(self, filename: str) -> str:
        """
        Save the model checkpoint to the given file name.
        Then save the checkpoint file path to the latest_ckpt file.
        """
        # Save the checkpoint file itself
        ckpt_dir = os.path.join(
            self.config.trainer.out_dir,
            self.config.trainer.experiment_name,
        )
        ckpt_path = os.path.join(ckpt_dir, filename)
        if len(self.lr_schedulers) == 1:
            self.model.lr_scheduler_state = self.lr_schedulers[0].state_dict()
        else:
            self.model.lr_scheduler_state = [s.state_dict() for s in self.lr_schedulers]
        self.model.save_checkpoint(ckpt_path)

        # Save the file path to the latest checkpoint
        if self.fabric.global_rank == 0:
            with open(
                os.path.join(self.config.trainer.out_dir, "latest_ckpt"),
                "w",
            ) as f:
                f.write(ckpt_path)

        return ckpt_path

    def _save_recovery_checkpoint(self):
        """
        Save a checkpoint for recovery from crashes.
        """
        # Save the checkpoint file
        ckpt_dir = os.path.join(
            self.config.trainer.out_dir,
            self.config.trainer.experiment_name,
        )
        ckpt_path = os.path.join(ckpt_dir, f"recovery_ckpt_iter_{self.step}.pt")
        if len(self.lr_schedulers) == 1:
            self.model.lr_scheduler_state = self.lr_schedulers[0].state_dict()
        else:
            self.model.lr_scheduler_state = [s.state_dict() for s in self.lr_schedulers]
        self.model.save_checkpoint(ckpt_path)

        # Save the most recent file path to the recovery checkpoint pointer file
        if self.fabric.global_rank == 0:
            with open(
                os.path.join(self.config.trainer.out_dir, "recovery_ckpt"),
                "w",
            ) as f:
                f.write(ckpt_path)

        # Delete the old recovery checkpoint file if it exists
        if self.recovery_checkpoint_path is not None:
            if self.fabric.global_rank == 0:
                os.remove(self.recovery_checkpoint_path)

        # Update the recovery checkpoint path
        self.recovery_checkpoint_path = ckpt_path

    def _select_lr_lambda(self) -> Callable[[Optional[int]], float]:
        schedule = str(self.config.lr_scheduler.schedule).lower()
        if schedule == "cosine":
            self.fabric.print("[LR Scheduler] using linear warmup + cosine decay")
            return self._get_lr_cosine
        elif schedule == "wsd":  # wsd
            self.fabric.print("[LR Scheduler] using warmup-stable-decay")
            return self._get_lr_warm_stable_decay
        else:  # constant
            self.fabric.print("[LR Scheduler] using constant learning rate")
            return lambda _: 1.0

    def _get_lr_warm_stable_decay(self, iter_num: Optional[int] = None) -> float:
        """
        LambdaLR callback for warmup -> stable -> linear warmdown.
        If warmup_iters == 0 and warmdown_iters == 0, this naturally becomes constant LR.
        """
        if iter_num is None:
            iter_num = self.step
        warmup_iters = self.config.lr_scheduler.get("warmup_iters", 0)
        warmdown_iters = self.config.lr_scheduler.get("warmdown_iters", 0)
        train_batches = self.config.trainer.train_batches

        # Constant LR mode when both warmup and warmdown are disabled.
        if warmup_iters <= 0 and warmdown_iters <= 0:
            return 1.0

        # 1) linear warmup to base LR
        if iter_num < warmup_iters:
            return iter_num / max(warmup_iters, 1)

        # 2) hold at base LR if warmdown is disabled.
        if not (warmdown_iters and warmdown_iters > 0):
            return 1.0

        # 3) hold until warmdown starts, then linearly warm down to 0.
        warmdown_start = max(warmup_iters, train_batches - warmdown_iters)
        if iter_num < warmdown_start:
            return 1.0
        if iter_num <= train_batches:
            warmdown_progress = iter_num - warmdown_start
            warmdown_ratio = warmdown_progress / max(warmdown_iters, 1)
            warmdown_ratio = min(max(warmdown_ratio, 0.0), 1.0)
            return 1.0 - warmdown_ratio
        return 0.0

    def _get_lr_cosine(self, iter_num: Optional[int] = None) -> float:
        """
        Cosine learning rate decay with warmup.
        Uses train_batches as the decay horizon (no separate decay_iters).
        Returns a multiplier for LambdaLR.
        """
        if iter_num is None:
            iter_num = self.step
        warmup_iters = self.config.lr_scheduler.get("warmup_iters", 0)
        decay_iters = self.config.trainer.train_batches
        max_lr = float(self.config.optimizer.learning_rate)
        min_lr = float(self.config.lr_scheduler.get("min_lr", 0.0))
        min_lr_ratio = min(max(min_lr / max(max_lr, 1e-12), 0.0), 1.0)

        # 1) linear warmup for warmup_iters steps.
        if warmup_iters > 0 and iter_num < warmup_iters:
            return iter_num / max(warmup_iters, 1)

        # 3) in between, use cosine decay down to min learning rate.
        denom = max(decay_iters - warmup_iters, 1)
        decay_ratio = (iter_num - warmup_iters) / denom
        decay_ratio = min(max(decay_ratio, 0.0), 1.0)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return min_lr_ratio + coeff * (1.0 - min_lr_ratio)
