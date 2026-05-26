import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from lightning.fabric import is_wrapped
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

from utils.import_utils import is_liger_kernel_available

LigerFusedLinearCrossEntropyLoss = None
if is_liger_kernel_available():
    from liger_kernel.transformers.fused_linear_cross_entropy import (
        LigerFusedLinearCrossEntropyLoss,
    )


class ModelBase:
    """Shared base class for BST and GPT models"""

    def __init__(self):
        # To be filled in by setup_fabric()
        self.fabric: L.Fabric = None

        # To be filled in by configure_optimizers()
        self.optimizer: Union[
            torch.optim.Optimizer, List[torch.optim.Optimizer], None
        ] = None

        # Total number of training steps
        self.training_steps: int = 0
        # Optional trainer-owned scheduler state persisted with checkpoints.
        self.lr_scheduler_state: Optional[Dict[str, Any]] = None

    def _assert_fabric_is_setup(self, setup: bool = True):
        """Checks that setup_fabric() has been called"""
        if setup:
            assert (
                self.fabric is not None
            ), "Fabric must be set up before calling this function"
        else:
            assert (
                self.fabric is None
            ), "This function must be called before setup_fabric()"

    def setup_fabric(self, fabric: L.Fabric):
        """
        Sets up the model with the provided Fabric object.
        This should set self.fabric = fabric and call fabric.setup_module()
        and fabric.setup_optimizers() on the applicable model and optimizer.
        """
        raise NotImplementedError

    def configure_optimizers(
        self,
        weight_decay: float,
        learning_rate: float,
        betas: Tuple[float, float],
        use_fused: Optional[bool] = None,
        optimizer_type: str = "adam",
        mu: float = 0.95,
        nesterov: bool = True,
        adjust_lr_fn: Optional[str] = None,
    ):
        """Create optimizer for training the model"""
        # model_*.py must indicate the modules whose parameters are optimized by configure_optimizers()
        # via self._optimizer_modules()
        param_dict = self._get_trainable_param_dict(*self._optimizer_modules())
        optimizer_type = optimizer_type.lower()
        # create optim groups based on choice of optimizer type
        if optimizer_type == "adam":
            self._configure_adamw(
                param_dict=param_dict,
                weight_decay=weight_decay,
                learning_rate=learning_rate,
                betas=betas,
                use_fused=use_fused,
            )
            return
        if optimizer_type == "muon":
            self._configure_muon_with_adamw(
                param_dict=param_dict,
                weight_decay=weight_decay,
                learning_rate=learning_rate,
                betas=betas,
                use_fused=use_fused,
                mu=mu,
                nesterov=nesterov,
                adjust_lr_fn=adjust_lr_fn,
            )
            return
        raise ValueError(f"Unsupported optimizer_type: {optimizer_type}")

    def compile(self):
        """Compiles the model for faster training and inference"""
        raise NotImplementedError

    def train(self):
        """Set the model to training mode"""
        raise NotImplementedError

    def eval(self):
        """Set the model to evaluation mode"""
        raise NotImplementedError

    def get_num_params(self, non_embedding: bool = True) -> int:
        """Returns the number of parameters in the model"""
        raise NotImplementedError

    def compute_loss(
        self,
        batch: torch.Tensor,  # Shape is (batch_size, seq_len)
        targets: torch.Tensor,  # Shape is (batch_size, seq_len)
        backpropagate: bool,  # Run backward pass if true, otherwise only compute loss
        no_sync: bool = False,  # If True, don't sync gradients across multiple GPUs
        loss_div: int = 1,  # Loss will be divided by this number
        **kwargs,  # Extra arguments ignored for compatibility between models
    ) -> torch.Tensor:
        """Computes the loss for a batch of data"""
        raise NotImplementedError

    def optimizer_step(
        self,
        grad_clip: Optional[float] = None,  # If given, clip gradient norm
    ) -> Optional[torch.Tensor]:
        """
        Performs a single optimization step.
        If gradient clipping is enabled, return the gradient norm before clipping.
        """
        # LR scheduling is handled in trainer via scheduler step.
        return self._optimizer_step_common(
            grad_clip=grad_clip,
            clip_modules=self._clip_modules(),
        )

    def _optimizer_modules(self) -> Tuple[nn.Module, ...]:
        """Modules whose parameters are optimized by configure_optimizers()."""
        raise NotImplementedError

    def _is_scalar_optimizer_param(self, name: str, param: torch.nn.Parameter) -> bool:
        """
        Returns True if parameter should be optimized by the scalar AdamW optimizer
        when using optimizer_type='muon'.
        """
        scalar_param_names = ("lm_head", "token_embedding")
        return param.ndim < 2 or any(tag in name for tag in scalar_param_names)

    def _clip_modules(self) -> Tuple[nn.Module, ...]:
        """Modules whose gradients are clipped by optimizer_step()."""
        return self._optimizer_modules()

    def _get_param_lr_overrides(
        self,
        param_dict: Dict[str, torch.nn.Parameter],
        learning_rate: float,
    ) -> Dict[str, float]:
        """
        Optional parameter-specific learning-rate overrides.
        Keys are parameter names from `param_dict`, values are absolute LRs.
        """
        return {}

    def _get_param_weight_decay_overrides(
        self,
        param_dict: Dict[str, torch.nn.Parameter],
        weight_decay: float,
    ) -> Dict[str, float]:
        """
        Optional parameter-specific weight-decay overrides.
        Keys are parameter names from `param_dict`, values are absolute weight decays.
        """
        return {}

    def _get_trainable_param_dict(
        self,
        *modules: nn.Module,
    ) -> Dict[str, torch.nn.Parameter]:
        """
        Return all trainable parameters from one or more modules.
        Raises if parameter names overlap across modules.
        """
        param_dict: Dict[str, torch.nn.Parameter] = {}
        for module in modules:
            module_params = {
                pn: p for pn, p in module.named_parameters() if p.requires_grad
            }
            overlap = set(param_dict).intersection(module_params)
            assert not overlap, f"Overlapping parameter names found: {overlap}"
            param_dict.update(module_params)
        return param_dict

    def _build_decay_param_groups(
        self,
        param_dict: Dict[str, torch.nn.Parameter],
        weight_decay: float,
        learning_rate: float,
    ) -> list[Dict[str, Any]]:
        """
        Build two AdamW groups: decay for matrix-like tensors, no-decay otherwise.
        """
        lr_overrides = self._get_param_lr_overrides(param_dict, learning_rate)
        wd_overrides = self._get_param_weight_decay_overrides(param_dict, weight_decay)
        groups_by_key: Dict[Tuple[float, float], List[torch.nn.Parameter]] = {}
        for name, param in param_dict.items():
            lr = float(lr_overrides.get(name, learning_rate))
            # Any matrix parameter will be weighted decayed, otherwise no
            # i.e., all weight tensors in matmuls + embeddings decay, all biases and layernorms don't
            default_wd = weight_decay if param.dim() >= 2 else 0.0
            wd = float(wd_overrides.get(name, default_wd))
            key = (lr, wd)
            groups_by_key.setdefault(key, []).append(param)

        groups: List[Dict[str, Any]] = []
        for (lr, wd), params in groups_by_key.items():
            if params:
                groups.append({"params": params, "weight_decay": wd, "lr": lr})
        return groups

    def _configure_adamw(
        self,
        *,
        param_dict: Dict[str, torch.nn.Parameter],
        weight_decay: float,
        learning_rate: float,
        betas: Tuple[float, float],
        use_fused: Optional[bool] = None,
    ):
        """Configure and store an AdamW optimizer from a parameter dictionary."""
        optim_groups = self._build_decay_param_groups(
            param_dict=param_dict,
            weight_decay=weight_decay,
            learning_rate=learning_rate,
        )
        # create AdamW optimizer and use the fused version if available
        self.optimizer = torch.optim.AdamW(
            optim_groups,
            lr=learning_rate,
            betas=betas,
            eps=1e-8,
            fused=use_fused,
        )

    def _configure_muon_with_adamw(
        self,
        *,
        param_dict: Dict[str, torch.nn.Parameter],
        weight_decay: float,
        learning_rate: float,
        betas: Tuple[float, float],
        use_fused: Optional[bool] = None,
        mu: float = 0.95,
        nesterov: bool = True,
        adjust_lr_fn: Optional[str] = None,
    ):
        lr_overrides = self._get_param_lr_overrides(param_dict, learning_rate)
        # Muon only supports strictly 2D tensors.
        matrix_params = [
            p
            for n, p in param_dict.items()
            if p.ndim == 2 and not self._is_scalar_optimizer_param(n, p)
        ]
        scalar_params = [
            p
            for n, p in param_dict.items()
            if not (p.ndim == 2 and not self._is_scalar_optimizer_param(n, p))
        ]
        matrix_names = [
            n
            for n, p in param_dict.items()
            if p.ndim == 2 and not self._is_scalar_optimizer_param(n, p)
        ]
        scalar_names = [
            n
            for n, p in param_dict.items()
            if not (p.ndim == 2 and not self._is_scalar_optimizer_param(n, p))
        ]
        print(f"Scalar optimizer parameters: {scalar_names}")
        print(f"Matrix optimizer parameters: {matrix_names}")

        scalar_items = [
            (n, p)
            for n, p in param_dict.items()
            if not (p.ndim == 2 and not self._is_scalar_optimizer_param(n, p))
        ]
        matrix_items = [
            (n, p)
            for n, p in param_dict.items()
            if p.ndim == 2 and not self._is_scalar_optimizer_param(n, p)
        ]
        scalar_groups_by_lr: Dict[float, List[torch.nn.Parameter]] = {}
        for name, param in scalar_items:
            lr = float(lr_overrides.get(name, learning_rate))
            scalar_groups_by_lr.setdefault(lr, []).append(param)
        scalar_groups = [
            {"params": params, "lr": lr}
            for lr, params in scalar_groups_by_lr.items()
            if params
        ]

        matrix_groups_by_lr: Dict[float, List[torch.nn.Parameter]] = {}
        for name, param in matrix_items:
            lr = float(lr_overrides.get(name, learning_rate))
            matrix_groups_by_lr.setdefault(lr, []).append(param)
        matrix_groups = [
            {"params": params, "lr": lr}
            for lr, params in matrix_groups_by_lr.items()
            if params
        ]

        optimizers: List[torch.optim.Optimizer] = []
        if scalar_params:
            # create AdamW optimizer for scalar parameters
            # scalar parameters are not weighted decayed
            optimizers.append(
                torch.optim.AdamW(
                    scalar_groups,
                    lr=learning_rate,
                    betas=betas,
                    eps=1e-8,
                    fused=use_fused,
                    weight_decay=0.0,
                )
            )
        if matrix_params:
            # create Muon optimizer for matrix parameters
            # matrix parameters are weighted decayed
            # adjust_lr_fn=None | match_rms_adamw; None by default
            optimizers.append(
                torch.optim.Muon(
                    matrix_groups,
                    lr=learning_rate,
                    weight_decay=weight_decay,
                    momentum=mu,
                    nesterov=nesterov,
                    adjust_lr_fn=adjust_lr_fn,
                )
            )
        if not optimizers:
            raise ValueError(
                "No trainable parameters found for optimizer configuration."
            )
        self.optimizer = optimizers if len(optimizers) > 1 else optimizers[0]

    def _optimizer_step_common(
        self,
        *,
        grad_clip: Optional[float] = None,
        clip_modules: Optional[Iterable[nn.Module]] = None,
    ) -> Optional[torch.Tensor]:
        """
        Shared optimizer step logic.
        Clips gradients (if requested), then steps and zeroes optimizer.
        """
        self._assert_fabric_is_setup()
        assert (
            self.optimizer is not None
        ), "Optimizer must be set up before calling this function"
        optimizers = (
            list(self.optimizer)
            if isinstance(self.optimizer, (list, tuple))
            else [self.optimizer]
        )

        grad_norm = None
        if grad_clip is not None and clip_modules is not None:
            clip_values = []
            for optimizer in optimizers:
                for module in clip_modules:
                    clip_values.append(
                        self.fabric.clip_gradients(
                            module, optimizer, max_norm=grad_clip
                        )
                    )
            if len(clip_values) == 1:
                grad_norm = clip_values[0]
            elif len(clip_values) > 1:
                grad_norm = torch.nn.utils.get_total_norm(clip_values)

        for optimizer in optimizers:
            optimizer.step()
        for optimizer in optimizers:
            optimizer.zero_grad(set_to_none=True)
        self.training_steps += 1
        return grad_norm

    def _setup_optimizer_with_fabric(self, fabric: L.Fabric):
        if self.optimizer is None:
            return
        if isinstance(self.optimizer, (list, tuple)):
            optimizers = list(self.optimizer)
            if all(is_wrapped(o) for o in optimizers):
                return
            setup_result = fabric.setup_optimizers(*optimizers)
            if isinstance(setup_result, tuple):
                self.optimizer = list(setup_result)
            else:
                self.optimizer = [setup_result]
        else:
            if not is_wrapped(self.optimizer):
                self.optimizer = fabric.setup_optimizers(self.optimizer)

    def _get_checkpoint_state(self) -> Dict[str, Any]:
        """Get the state of the model for checkpoint save/load"""
        raise NotImplementedError

    def save_checkpoint(self, file_path: str):
        """
        Save the model checkpoint to the given file path.
        This includes the encoder, text head, optimizer, and training steps.
        """
        self._assert_fabric_is_setup()
        self.fabric.print(f"Saving checkpoint to {file_path}")
        state = self._get_checkpoint_state()
        if "training_steps" not in state:
            # Make sure to save training steps
            state["training_steps"] = self.training_steps
        if self.lr_scheduler_state is not None:
            state["lr_scheduler_state"] = self.lr_scheduler_state
        self.fabric.save(file_path, state)

    def load_checkpoint(self, file_path: str, strict: bool = True):
        """
        Load the model checkpoint from the given file path.
        If model does not have optimizer, load only the model weights.
        """
        self._assert_fabric_is_setup()
        self.fabric.print(f"Loading checkpoint from {file_path}")
        state = self._get_checkpoint_state()
        if self.optimizer is None:
            # Optimizer has not been initialized, so don't load it
            # No optimizer needed for inference
            self.fabric.print(
                "Optimizer not configured, loading model for inference only"
            )
            state.pop("optimizer", None)
        # fabric.load() will in-place modify all objects in the state
        self.fabric.load(file_path, state, strict=strict, weights_only=False)
        # Update training steps manually because it is an int
        self.training_steps = state["training_steps"]
        self.lr_scheduler_state = self._read_scheduler_state_from_checkpoint(file_path)

    @staticmethod
    def _read_scheduler_state_from_checkpoint(
        file_path: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Read optional lr_scheduler_state directly from checkpoint payload if present.
        """
        try:
            raw_state = torch.load(file_path, map_location="cpu", weights_only=False)
            if isinstance(raw_state, dict):
                scheduler_state = raw_state.get("lr_scheduler_state")
                if isinstance(scheduler_state, dict):
                    return scheduler_state
        except Exception:
            # Scheduler state is optional and checkpoint format may vary by strategy.
            return None
        return None

    def get_custom_metrics(self) -> Dict[str, Any]:
        """
        Returns a dictionary of custom metrics for the model.
        This is used for logging and monitoring during training.
        """
        return {}

    @torch.no_grad()
    def compute_hidden_state_rank(
        self, hidden_states: torch.Tensor, loss_div: int = 1
    ) -> dict:
        """
        Compute batch-level rank metrics for hidden states.

        Args:
            hidden_states: Tensor of shape (batch_size, seq_len, hidden_dim)

        Returns:
            Dictionary containing rank metrics
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape
        loss_div = max(int(loss_div), 1)

        # Compute rank for the entire batch (treating as one large matrix)
        try:
            # Reshape to (batch_size * seq_len, hidden_dim)
            batch_matrix = hidden_states.view(-1, hidden_dim)

            # Compute numerical rank using torch.linalg.matrix_rank
            atol = 1e-3
            rtol = 1e-3
            batch_numerical_rank = torch.linalg.matrix_rank(
                batch_matrix, atol=atol, rtol=rtol
            ).item()

            # Compute SVD for effective rank and condition number
            U_batch, S_batch, V_batch = torch.svd(batch_matrix)

            # Effective rank (Shannon entropy of normalized singular values)
            normalized_s_batch = S_batch / S_batch.sum()
            epsilon = 1e-12
            normalized_s_batch = torch.clamp(normalized_s_batch, min=epsilon)
            batch_effective_rank = torch.exp(
                -torch.sum(normalized_s_batch * torch.log(normalized_s_batch))
            ).item()

            # Condition number
            batch_condition_number = (
                (S_batch[0] / S_batch[-1]).item() if S_batch[0] > 0 else float("inf")
            )

            # Maximum possible rank
            max_possible_rank = min(batch_size * seq_len, hidden_dim)

            rank_stats = {
                "batch_numerical_rank": torch.tensor(
                    batch_numerical_rank,
                    device=hidden_states.device,
                    dtype=torch.float32,
                ),
                "batch_effective_rank": torch.tensor(
                    batch_effective_rank,
                    device=hidden_states.device,
                    dtype=torch.float32,
                ),
                "batch_condition_number": torch.tensor(
                    batch_condition_number,
                    device=hidden_states.device,
                    dtype=torch.float32,
                ),
                "max_possible_rank": torch.tensor(
                    max_possible_rank, device=hidden_states.device, dtype=torch.float32
                ),
                "rank_utilization": torch.tensor(
                    batch_numerical_rank / max_possible_rank,
                    device=hidden_states.device,
                    dtype=torch.float32,
                ),
            }

        except Exception as e:
            rank_stats = {
                "batch_numerical_rank": torch.tensor(
                    0.0, device=hidden_states.device, dtype=torch.float32
                ),
                "batch_effective_rank": torch.tensor(
                    0.0, device=hidden_states.device, dtype=torch.float32
                ),
                "batch_condition_number": torch.tensor(
                    float("inf"), device=hidden_states.device, dtype=torch.float32
                ),
                "max_possible_rank": torch.tensor(
                    min(batch_size * seq_len, hidden_dim),
                    device=hidden_states.device,
                    dtype=torch.float32,
                ),
                "rank_utilization": torch.tensor(
                    0.0, device=hidden_states.device, dtype=torch.float32
                ),
            }

        return {k: (v / loss_div) for k, v in rank_stats.items()}


class DocumentRelativePositions:
    """
    Shared code between BST and GPT models for handling packed sequences.
    This is used for computing positions of each token relative to each document.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @torch.no_grad()
    def create_position_indices(self, batch: torch.Tensor) -> torch.Tensor:
        """
        Create document-relative position indices for a batch of token sequences.
        Position indices restart for each document, and a sequence can have multiple packed documents.

        In other words, the indices are absolute within a document but relative to each
        document across the entire sequence. EOS tokens always have position index 0.

        For example
            Sequence: [A, B, C, EOS, D, E, F, G, H, EOS, I, J]
            Result:   [1, 2, 3,  0,  1, 2, 3, 4, 5,  0,  1, 2]
        """
        assert self.config is not None, "Expected super-class to set self.config"
        assert self.config.eos_token_id is not None, "self.config.eos_token_id is None"

        batch_size, seq_len = batch.shape
        device = batch.device

        # Find positions of EOS tokens
        # Sequence: [A, B, C, EOS, D, E, F, G, H, EOS, I, J]
        # EOS:      [0, 0, 0,  1,  0, 0, 0, 0, 0,  1,  0, 0]
        eos_positions = batch == self.config.eos_token_id

        # Create indices relative to entire sequence
        # Sequence:    [A, B, C, EOS, D, E, F, G, H, EOS, I, J]
        # seq_indices: [0, 1, 2,  3,  4, 5, 6, 7, 8,  9, 10, 11]
        seq_indices = (
            torch.arange(seq_len, device=device)
            .unsqueeze(0)
            .expand(batch_size, seq_len)
        )

        # Compute offset of each document relative to the sequence
        # Start with tensor filled with -1
        # Replace EOS positions with their index, and leave all other positions as -1
        # Sequence:    [ A,  B,  C, EOS,  D,  E,  F,  G,  H, EOS, I,  J]
        # doc_offsets: [-1, -1, -1,  3,  -1, -1, -1, -1, -1,  9, -1, -1]
        doc_offsets = torch.full(
            (batch_size, seq_len), fill_value=-1, device=device, dtype=seq_indices.dtype
        )
        doc_offsets[eos_positions] = seq_indices[eos_positions]

        # Take cumulative maximum to get the offset of each document
        # Sequence:    [ A,  B,  C, EOS, D, E, F, G, H, EOS, I, J]
        # doc_offsets: [-1, -1, -1,  3,  3, 3, 3, 3, 3,  9,  9, 9]
        doc_offsets = torch.cummax(doc_offsets, dim=1).values

        # To get document indices, subtract the offset from the sequence indices
        # If first token is not EOS, the offset is -1, so the document indices correctly starts at 1
        # Sequence:    [ A,  B,  C, EOS, D, E, F, G, H, EOS, I, J]
        # seq_indices: [ 0,  1,  2,  3,  4, 5, 6, 7, 8,  9, 10, 11]
        # doc_offsets: [-1, -1, -1,  3,  3, 3, 3, 3, 3,  9,  9, 9]
        # Result:      [ 1,  2,  3,  0,  1, 2, 3, 4, 5,  0,  1, 2]
        doc_indices = seq_indices - doc_offsets

        return doc_indices


class FusedCrossEntropyLoss:
    """
    Wrapper that combines the last linear layer with cross entropy loss.
    This enables the use of fused Liger kernel to avoid storing the logits tensor,
    which saves memory when the vocabulary size is large.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._liger_available = is_liger_kernel_available()
        self._fused_loss_fn = None
        if self._liger_available:
            self._fused_loss_fn = LigerFusedLinearCrossEntropyLoss(
                ignore_index=-100,
                reduction="mean",
            )

    def cross_entropy_loss(
        self,
        input: torch.Tensor,
        last_layer: nn.Linear,
        targets: torch.Tensor,
        reduction: str = "mean",
    ) -> torch.Tensor:
        """
        Compute result of passing input through last_layer
        and then applying cross-entropy loss with targets.

        This function should only be called in forward()
        """
        assert self.config is not None, "Expected super-class to set self.config"

        input_flat = input.reshape(-1, input.size(-1))
        targets_flat = targets.reshape(-1)
        assert input_flat.size(0) == targets_flat.size(
            0
        ), f"Flattened input and target shapes do not match: {input_flat.shape} vs {targets_flat.shape}"

        use_fused = self.config.use_fused and self._liger_available

        if use_fused:
            return self._fused_cross_entropy_loss(input_flat, last_layer, targets_flat)
        else:
            return self._standard_cross_entropy_loss(
                input_flat, last_layer, targets_flat, reduction
            )

    # Fused kernel causes problems with torch.compile
    @torch.compiler.disable(recursive=True)
    def _fused_cross_entropy_loss(
        self,
        input_flat: torch.Tensor,
        last_layer: nn.Linear,
        targets_flat: torch.Tensor,
    ) -> torch.Tensor:
        # Make sure tensors are contiguous
        input_flat = input_flat.contiguous()
        targets_flat = targets_flat.contiguous()
        # Manually type cast last_layer weights to match input dtype
        # This is needed to avoid issues with mixed precision
        weight = last_layer.weight.to(dtype=input_flat.dtype)
        bias = (
            last_layer.bias.to(dtype=input_flat.dtype)
            if last_layer.bias is not None
            else None
        )
        return self._fused_loss_fn(weight, input_flat, targets_flat, bias=bias)

    def _standard_cross_entropy_loss(
        self,
        input_flat: torch.Tensor,
        last_layer: nn.Linear,
        targets_flat: torch.Tensor,
        reduction: str = "mean",
    ) -> torch.Tensor:
        logits = last_layer(input_flat)
        return F.cross_entropy(logits, targets_flat, reduction=reduction)


class RotaryPositionEmbedding:
    def __init__(self, max_seq_len: int, head_dim: int, base_freq: float = 10000):
        assert head_dim % 2 == 0, "Dimension of attention head must be even"
        # round max_seq_len up to next multiple of 256
        self.max_seq_len = (max_seq_len + 255) // 256 * 256
        self.head_dim = head_dim
        self.base_freq = base_freq
        self.cos_lookup = None
        self.sin_lookup = None

    @torch.no_grad()
    def __call__(self, pos: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the cosine and sine components of the rotary position embedding.
        Input is a tensor of shape (batch_size, seq_len) containing position indices
        Returns cos and sin tensors of shape (batch_size, seq_len, head_dim)
        """
        assert (
            pos.max() < self.max_seq_len
        ), f"Position index {pos.max()} exceeds max_seq_len {self.max_seq_len}"

        if self.cos_lookup is None or self.sin_lookup is None:
            # Precompute cos and sin values if not already done
            self._precompute_cos_sin(pos.device)

        if self.cos_lookup.device != pos.device:
            # Move precomputed values to the same device as pos
            self.cos_lookup = self.cos_lookup.to(pos.device)
            self.sin_lookup = self.sin_lookup.to(pos.device)

        return self.cos_lookup[pos], self.sin_lookup[pos]

    def _precompute_cos_sin(self, device: torch.device):
        # Always use full precision for precomputation
        with torch.autocast(device_type=device.type, enabled=False):
            # Create frequencies of shape (head_dim/2)
            dims = torch.arange(
                0, self.head_dim, 2, dtype=torch.int64, device=device
            ).float()
            freqs = 1.0 / (self.base_freq ** (dims / self.head_dim))
            # Create position indices of shape (max_seq_len)
            positions = torch.arange(
                0, self.max_seq_len, dtype=torch.int64, device=device
            ).float()
            # Compute angles of shape (max_seq_len, head_dim/2)
            angles = torch.outer(positions, freqs)
            # Repeat angles to get shape (max_seq_len, head_dim)
            angles = torch.cat((angles, angles), dim=-1)
            # Compute cosine and sine lookup tables
            self.cos_lookup = angles.cos()
            self.sin_lookup = angles.sin()

    @staticmethod
    def apply(
        q: torch.Tensor,
        k: torch.Tensor,
        rope_cos_sin: Tuple[torch.Tensor, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply rotary position embedding to query and key tensors.

        q and k have shape (batch_size, num_heads, seq_len, head_dim)
        cos and sin have shape (batch_size, seq_len, head_dim)

        If all sequences in the batch have the same positions,
        shape of cos and sin can also be (1, seq_len, head_dim)
        """
        cos, sin = rope_cos_sin
        assert cos is not None and sin is not None
        assert cos.shape == sin.shape

        # Unsqueeze cos and sin to match the dimensions of q and k
        # Shape becomes (batch_size, 1, seq_len, head_dim)
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)

        # Apply rotations
        q_rot = q * cos + RotaryPositionEmbedding._rotate_half(q) * sin
        k_rot = k * cos + RotaryPositionEmbedding._rotate_half(k) * sin

        return q_rot, k_rot

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        """
        Rotates half the hidden dims of the input.
        This is non-interleaved Llama / Huggingface style rotation:
            [1, 2, 3, 4, 5, 6, 7, 8] -> [-5, -6, -7, -8, 1, 2, 3, 4]
        https://github.com/huggingface/transformers/blob/v4.50.0/src/transformers/models/llama/modeling_llama.py#L151
        """
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)


class LayerNorm(nn.Module):
    """LayerNorm but with an optional bias. If bias=False, use RMSNorm."""

    def __init__(
        self,
        ndim: int,
        bias: bool,
        eps: float = 1e-5,
    ):
        super().__init__()
        self.eps = eps

        if bias:
            self.weight = nn.Parameter(torch.ones(ndim))
            self.bias = nn.Parameter(torch.zeros(ndim))
        else:
            self.weight = nn.Parameter(torch.ones(ndim))
            self.register_parameter("bias", None)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.bias is not None:
            return F.layer_norm(
                input, self.weight.shape, self.weight, self.bias, self.eps
            )
        else:
            # just use RMSNorm
            return F.rms_norm(input, self.weight.shape, self.weight, self.eps)


class SwiGLU(nn.Module):
    """Linear layer with SwiGLU activation function"""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        super().__init__()
        self.gate_up = nn.Linear(input_size, 2 * output_size, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Split the linear layer output into two chunks
        gate_up = self.gate_up(x)
        gate, up = gate_up.chunk(2, dim=-1)
        return F.silu(gate) * up
