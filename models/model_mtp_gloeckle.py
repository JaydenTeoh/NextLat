"""
Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import itertools
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from dataclasses import dataclass
from lightning.fabric import is_wrapped
from torch.distributed.fsdp import FSDPModule
from typing import Any, Dict, List, Optional, Tuple

from models.model_base import (
    ModelBase,
    DocumentRelativePositions,
    FusedCrossEntropyLoss,
    LayerNorm,
    RotaryPositionEmbedding,
    SwiGLU,
)
from models.model_speculative import SpeculativeModel
from utils.speculative_sampling import normalize_logits, sample_from_probs
from models.model_gpt import GPTConfig, Block


@dataclass
class MTPGloeckleConfig:
    block_size: int = 1024
    # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    bias: bool = False
    context_length: int = 0  # default is zero
    eos_token_id: int = -1
    compute_hidden_state_rank: bool = False
    # True: use fused Liger kernels. False: use regular PyTorch functions.
    use_fused: bool = False
    # MTP parameters
    mtp_horizon: int = 1  # multi-step prediction horizon (d) in the paper
    mtp_lambda: float = 1.0  # weight for the MTP loss


class MTPTransformer(DocumentRelativePositions, FusedCrossEntropyLoss, nn.Module):
    def __init__(self, config: GPTConfig, mtp_horizon: int):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        # shared for forward and backward encoders
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)

        # add 1 extra position embedding for implicit EOS at start
        n_pos_embd = config.block_size + 1
        self.rotary_embedding = RotaryPositionEmbedding(
            max_seq_len=n_pos_embd,
            head_dim=config.n_embd // config.n_head,
        )

        self.shared_trunk = nn.ModuleDict(
            dict(
                # minus one layer because we add another layer for the MTP heads
                blocks=nn.ModuleList(
                    [Block(config) for _ in range(config.n_layer - 1)]
                ),
            )
        )
        self.mtp_horizon = mtp_horizon
        assert mtp_horizon >= 0, "MTP depth cannot be negative"
        self.mtp_heads = nn.ModuleList()
        for i in range(mtp_horizon + 1):
            self.mtp_heads.append(
                nn.ModuleDict(
                    dict(
                        block=Block(config),
                        norm=LayerNorm(config.n_embd, bias=config.bias),
                    )
                )
            )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # init all weights
        self.apply(self._init_weights)

    def get_num_params(self, non_embedding=True) -> int:
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the token embeddings get subtracted.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.token_embedding.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    @torch.no_grad()
    def create_attention_mask(self, batch: torch.Tensor) -> torch.Tensor:
        """
        Create causal attention mask for a sequence of packed documents.
        Mask will avoid attending to tokens in different documents.
        """
        # batch has shape (batch_size, seq_len)

        # find positions of eos tokens
        # shape is (batch_size, seq_len)
        eos_positions = batch == self.config.eos_token_id

        # create indices of packed documents in the token sequence
        # each token can attend up to and including previous EOS, but not next EOS
        # shape is (batch_size, seq_len)
        document_id = torch.cumsum(eos_positions, dim=1)

        # create mask of tokens within the same document
        # shape is (batch_size, seq_len, seq_len)
        document_mask = document_id.unsqueeze(1) == document_id.unsqueeze(2)

        # lower triangular causal mask
        mask = torch.tril(document_mask)

        return mask

    def forward(
        self,
        batch: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        targets: Optional[torch.Tensor] = None,
        return_hidden_states: bool = False,
        next_token_only: bool = True,
    ) -> torch.Tensor:
        batch_size, seq_len = batch.size()
        assert (
            seq_len <= self.config.block_size
        ), f"Cannot forward sequence of length {seq_len}, block size is only {self.config.block_size}"

        # token positions relative to each document
        pos = self.create_position_indices(batch)
        rope = self.rotary_embedding(pos)

        if mask is None:
            # causal attention mask for packed sequence
            mask = self.create_attention_mask(batch)

        # token embeddings of shape (b, t, n_embd)
        x = self.token_embedding(batch)

        for block in self.shared_trunk.blocks:
            x = block(x, mask=mask, rope=rope)

        if not next_token_only:  # multi-token prediction
            # accumulate gradients for the MTP heads
            x_det = x.detach()
            x_det.requires_grad_()
            logits = None
            losses = None
            # Precompute document IDs once; reused by all shifted heads.
            eos_positions = batch == self.config.eos_token_id
            doc_ids = torch.cumsum(eos_positions, dim=1)
            if targets is None:
                logits = x_det.new_empty(
                    (batch_size, seq_len, self.mtp_horizon + 1, self.config.vocab_size)
                )
            else:
                losses_list = []
            for i, mtp_head in enumerate(self.mtp_heads):
                x_i = self.mtp_heads[i].block(x_det, mask=mask, rope=rope)
                x_i = self.mtp_heads[i].norm(x_i)  # (batch_size, seq_len, n_embd)
                # fix document crossing: mask predictions that would cross document boundaries
                if targets is not None:
                    if i == 0:
                        # No shift for i=0
                        input_slice = x_i
                        target_slice = targets
                    else:
                        input_slice = x_i[:, :-i]
                        target_slice = targets[:, i:]

                        # Get document IDs for current positions and target positions
                        current_doc_ids = doc_ids[
                            :, :-i
                        ]  # Document IDs at current positions
                        target_doc_ids = doc_ids[
                            :, i:
                        ]  # Document IDs at target positions

                        # We do not skip predictions at EOS tokens.
                        is_eos_target = target_slice == self.config.eos_token_id
                        adjusted_target_doc_ids = target_doc_ids - is_eos_target.long()

                        # Check if current and target positions are in different documents
                        cross_boundary_mask = current_doc_ids != adjusted_target_doc_ids

                        # Mask out target positions that cross document boundaries
                        target_slice = target_slice.masked_fill(
                            cross_boundary_mask, -100
                        )

                    losses_list.append(
                        self.cross_entropy_loss(
                            input=input_slice,
                            last_layer=self.lm_head,
                            targets=target_slice,
                        )
                    )
                else:
                    logits[:, :, i, :] = self.lm_head(x_i)

            if targets is not None:
                losses = torch.stack(losses_list, dim=0)
            output = (
                logits,
                losses,
                x_det,
                x,
            )  # x_det and x are used for gradient accumulation
        else:  # next token only
            x_1 = self.mtp_heads[0].block(x, mask=mask, rope=rope)
            x_1 = self.mtp_heads[0].norm(x_1)  # (batch_size, seq_len, n_embd)
            logits = self.lm_head(x_1)

            output = logits

        if return_hidden_states:
            return output, x

        return output

    def crop_block_size(self, block_size: int):
        # model surgery to decrease the block size if necessary
        # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        for block in self.transformer.blocks:
            if hasattr(block.attn, "bias"):
                block.attn.bias = block.attn.bias[:, :, :block_size, :block_size]


class MTPGloeckle(ModelBase, SpeculativeModel):
    def __init__(
        self,
        config: MTPGloeckleConfig,
    ):
        super().__init__()

        self.config = config
        # the base GPT
        base_config = GPTConfig(
            block_size=config.block_size,
            vocab_size=config.vocab_size,
            n_layer=config.n_layer,
            n_head=config.n_head,
            n_embd=config.n_embd,
            dropout=config.dropout,
            bias=config.bias,
        )
        self.model = MTPTransformer(base_config, config.mtp_horizon)

    def setup_fabric(self, fabric: L.Fabric):
        """
        Setup Lightning Fabric for distributed training
        This wraps the models and optimizer with a FabricModule
        """
        self.fabric = fabric

        if not is_wrapped(self.model):
            self.model = fabric.setup_module(self.model)
            # Print model architecture
            self.fabric.print(self.model)
            self.fabric.print(f"Total number of parameters: {self.get_num_params():,}")

        self._setup_optimizer_with_fabric(fabric)

    def compute_loss(
        self,
        batch: torch.Tensor,  # Shape is (batch_size, seq_len)
        targets: torch.Tensor,  # Shape is (batch_size, seq_len-1)
        backpropagate: bool,  # Run backward pass if true, otherwise only compute loss
        no_sync: bool = False,  # If True, don't sync gradients across multiple GPUs
        loss_div: int = 1,  # Loss will be divided by this number
        **kwargs,  # Extra arguments ignored for compatibility with BST
    ) -> torch.Tensor:
        self._assert_fabric_is_setup()

        # Predict next token for all tokens before the last token
        if targets.shape[-1] != batch.shape[-1]:
            inputs = batch[:, :-1]
        else:
            inputs = batch  # for a5 dataset

        if self.config.context_length > 0:
            # Create context mask: mask positions less than context_length
            # Sequence:    [ A,  B,  C, EOS, D, E, F, G, H, EOS, I, J]
            # pos_ids:     [ 1,  2,  3,  0,  1, 2, 3, 4, 5,  0,  1, 2]
            pos_ids = self.model.create_position_indices(targets)
            context_length_mask = (pos_ids <= self.config.context_length + 1) & (
                pos_ids != 0
            )
            targets = targets.masked_fill(context_length_mask, -100)

        # If using FSDP, we ignore no_sync and always sync gradients
        # This allows us to re-shard each layer of the model after backward pass
        is_fsdp = isinstance(self.model.module, FSDPModule)
        with self.fabric.no_backward_sync(self.model, no_sync and not is_fsdp):
            # Get losses over all MTP heads
            logits, losses, x_det, x = self.model(
                inputs, targets=targets, next_token_only=False
            )

            losses = losses / loss_div
            if losses.numel() > 1:
                total_loss = losses[0] + self.config.mtp_lambda * torch.mean(losses[1:])
            else:
                total_loss = losses[0]

            # Backward pass with gradient accumulation
            if backpropagate:
                self.fabric.backward(total_loss)
                hidden_states_grad = (
                    x_det.grad if x_det.grad is not None else torch.zeros_like(x_det)
                )
                self.fabric.backward(x, gradient=hidden_states_grad)

        result = {
            "loss": total_loss.detach(),
            "next_token_loss": losses[0].detach(),
        }
        result.update(
            {
                f"loss_token_{i+1}": losses[i + 1].detach()
                for i in range(self.config.mtp_horizon)
            }
        )

        if self.config.compute_hidden_state_rank:
            hidden_states_det = x.detach()
            rank_stats = self.compute_hidden_state_rank(
                hidden_states_det, loss_div=loss_div
            )
            result.update({f"{k}": v for k, v in rank_stats.items()})

        return result

    def _optimizer_modules(self) -> Tuple[nn.Module, ...]:
        return (self.model,)

    def compile(self):
        """
        Compiles the model using torch.compile()
        """
        # Must compile before setup_fabric()
        self._assert_fabric_is_setup(setup=False)
        self.model.compile()

    def train(self):
        """
        Set the model to training mode
        """
        self.model.train()

    def eval(self):
        """
        Set the model to evaluation mode
        """
        self.model.eval()

    def get_num_params(self, non_embedding: bool = True) -> int:
        return self.model.get_num_params(non_embedding)

    def _get_checkpoint_state(self) -> Dict[str, Any]:
        """
        Get the state of the model for checkpoint save/load
        """
        return {
            "model": self.model,
            "optimizer": self.optimizer,
            "training_steps": self.training_steps,
        }

    def crop_block_size(self, block_size: int):
        self.model.crop_block_size(block_size)

    def estimate_mfu(self, fwdbwd_per_iter: int, dt: float) -> float:
        """estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS"""
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd // cfg.n_head, cfg.block_size
        flops_per_token = 6 * N + 12 * L * H * Q * T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0 / dt)  # per second
        flops_promised = 312e12  # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.inference_mode()
    def speculative_propose(
        self,
        seq: torch.Tensor,
        steps_to_propose: int,
        temperature: float,
        top_k: Optional[int],
        top_p: Optional[float],
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        core = self._unwrap_module()
        all_outputs = core(seq, next_token_only=False)
        all_logits = all_outputs[0]
        if not isinstance(all_logits, torch.Tensor) or all_logits.ndim != 4:
            return self._propose_autoregressive(
                seq,
                steps_to_propose=steps_to_propose,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )

        horizon_logits = all_logits[:, -1, :, :]
        drafted: List[torch.Tensor] = []
        q_probs_steps: List[torch.Tensor] = []
        max_head = horizon_logits.size(1) - 1
        for i in range(steps_to_propose):
            head_idx = min(i, max_head)
            q_probs = normalize_logits(
                horizon_logits[:, head_idx, :],
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )
            tok = sample_from_probs(q_probs)
            drafted.append(tok)
            q_probs_steps.append(q_probs)
        return torch.cat(drafted, dim=1), q_probs_steps

    @torch.inference_mode()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = (
                idx
                if idx.size(1) <= self.config.block_size
                else idx[:, -self.config.block_size :]
            )
            # forward the model to get the logits for the index in the sequence
            logits = self.model(idx_cond, next_token_only=True)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx

    @torch.inference_mode()
    def evaluation_loss(
        self,
        batch: torch.Tensor,  # Shape is (batch_size, seq_len)
        prefix_end_index: torch.Tensor,  # Shape is (batch_size)
        suffix_start_index: torch.Tensor,  # Shape is (batch_size)
    ) -> torch.Tensor:
        """
        Compute next token prediction loss on the given batch of sequences.
        Prompt tokens are excluded from loss computation.
        """
        batch_size, seq_len = batch.size()

        # Sequences in batch without prefix/suffix
        # Shape is (batch_size)
        no_prefix = prefix_end_index == -1
        no_suffix = suffix_start_index == -1

        # Get the start and end indices of the generated portion of the sequence
        # Shape is (batch_size)
        gen_start_index = prefix_end_index + 1
        gen_end_index = suffix_start_index - 1
        gen_start_index[no_prefix] = 0
        gen_end_index[no_suffix] = seq_len - 2

        # Forward pass
        logits = self.model(batch, next_token_only=True)

        # Compute loss for each sequence in batch
        # Use single sequence to compare against BST model
        next_token_loss = torch.zeros(batch_size, device=batch.device)
        for i in range(batch_size):
            # Get the target tokens for this sequence
            # Indices are inclusive of start and end
            targets = batch[i, gen_start_index[i] + 1 : gen_end_index[i] + 2]
            logits_batch = logits[i, gen_start_index[i] : gen_end_index[i] + 1]

            loss = F.cross_entropy(
                logits_batch.view(-1, logits_batch.size(-1)),
                targets.view(-1),
            )

            next_token_loss[i] = loss

        # Zeroes for second return value because no previous token prediction loss
        return next_token_loss, torch.zeros(batch_size, device=batch.device)
