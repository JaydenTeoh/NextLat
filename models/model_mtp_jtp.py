"""
Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from dataclasses import dataclass
from lightning.fabric import is_wrapped
from torch.distributed.fsdp import FSDPModule
from typing import Any, Dict, Optional, Tuple, List

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
class MTPJTPConfig:
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
    mtp_lambda: float = 1.0  # loss weight for the JTP loss


def build_teacher_and_target_idx(
    idx: torch.Tensor,
    B: int,
    T: int,
    D: int,
    context_length: int,
    pos_ids: torch.Tensor,
    eos_token_id: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns teacher_idx and target_idx tensors, both shape (B, T, D).
    * teacher_idx[b, i, k] = idx[b, i+k] if valid; else 0
    * target_idx[b, i, k]  = idx[b, i+k+1] if valid; else -100
    Also zeros out the first `context_length` positions for teacher_idx
    and sets them to -100 for target_idx to skip MTP training there.
    """
    positions = torch.arange(T, device=device).unsqueeze(0).unsqueeze(-1)  # (1, T, 1)
    # teacher: idx[b, i+k]
    shifts_teacher = torch.arange(1, D + 1, device=device).view(1, 1, D)
    teacher_positions = positions + shifts_teacher
    teacher_positions = teacher_positions.expand(B, T, D)
    teacher_mask = teacher_positions < T

    teacher_positions_clamped = teacher_positions.clone().clamp_(0, T - 1)
    teacher_idx_temp = torch.gather(
        idx.unsqueeze(-1).expand(B, T, D), dim=1, index=teacher_positions_clamped
    )
    teacher_idx = torch.where(
        teacher_mask, teacher_idx_temp, torch.zeros_like(teacher_idx_temp)
    )

    # target: idx[b, i+k+1]
    shifts_target = torch.arange(2, D + 2, device=device).view(1, 1, D)
    target_positions = positions + shifts_target
    target_positions = target_positions.expand(B, T, D)
    target_mask = target_positions < T

    target_positions_clamped = target_positions.clone().clamp_(0, T - 1)
    target_idx_temp = torch.gather(
        idx.unsqueeze(-1).expand(B, T, D), dim=1, index=target_positions_clamped
    )
    target_idx = torch.where(
        target_mask, target_idx_temp, torch.full_like(target_idx_temp, -100)
    )

    # fix document crossing: if a teacher token is EOS, then all subsequent tokens (for that position)
    # cross into a new document. So, mask those out in the target.
    eos_mask = teacher_idx == eos_token_id
    target_eos_mask = torch.cummax(eos_mask, dim=-1).values
    target_idx[target_eos_mask] = -100

    if context_length > 0:
        # fix positions for context length masking
        # Sequence:    [ A,  B,  C, EOS, D, E, F, G, H, EOS, I, J]
        # pos_ids:     [ 1,  2,  3,  0,  1, 2, 3, 4, 5,  0,  1, 2]
        pos_ids = pos_ids.unsqueeze(-1)  # (B, T, 1)
        teacher_pos_ids = pos_ids + shifts_teacher  # (B, T, D)
        context_length_mask = (teacher_pos_ids <= context_length + 1) & (
            teacher_pos_ids != 0
        )
        teacher_idx = teacher_idx.masked_fill(context_length_mask, 0)
        target_idx = target_idx.masked_fill(context_length_mask, -100)

    return teacher_idx, target_idx


class RelativeSelfAttention(nn.Module):
    """
    Relative self-attention layer that incorporates learned relative positional
    bias into the attention computation.
    """

    def __init__(self, config: GPTConfig, max_len: int):
        super().__init__()
        assert (
            config.n_embd % config.n_head == 0
        ), "Embedding dim must be divisible by number of heads"
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.max_len = max_len

        # QKV projection
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # Output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        # Learned relative positional bias of shape (n_head, max_len, max_len)
        self.rel_pos_bias = nn.Parameter(torch.zeros(self.n_head, max_len, max_len))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, d_model)
        Returns:
            y: (B, T, d_model)
        """
        B, T, C = x.size()
        qkv = self.c_attn(x)  # (B, T, 3*C)
        q, k, v = qkv.split(C, dim=2)
        # reshape to (B, n_head, T, head_dim)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        att = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(
            self.head_dim
        )  # (B, n_head, T, T)
        mask = torch.tril(torch.ones(T, T, device=x.device)).unsqueeze(0).unsqueeze(0)
        att = att.masked_fill(mask == 0, float("-inf"))

        # Add relative positional bias.
        if T < self.max_len:
            rel_bias = self.rel_pos_bias[:, :T, :T]
        else:
            rel_bias = self.rel_pos_bias
        att = att + rel_bias.unsqueeze(0)

        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = torch.matmul(att, v)
        # Reassemble heads
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y


# Modified RelativeBlock without extra LayerNorm
class RelativeBlock(nn.Module):
    """
    Relative attention block that wraps the RelativeSelfAttention without additional LayerNorm,
    to be equivalent with Implementation A.
    """

    def __init__(self, config, max_len: int):
        super().__init__()
        self.attn = RelativeSelfAttention(config, max_len=max_len)

    def forward(self, x):
        return self.attn(x)


# TeacherCombine module equivalent to ln_teacher(e + gamma * h)
class TeacherCombine(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.gamma = nn.Parameter(torch.tensor(0.1))
        self.ln = LayerNorm(d_model, bias=True)

    def forward(self, h: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        return self.ln(emb + self.gamma * h)


class JTPTransformer(DocumentRelativePositions, FusedCrossEntropyLoss, nn.Module):
    def __init__(self, config: MTPJTPConfig):
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

        self.transformer = nn.ModuleDict(
            dict(
                blocks=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                norm=LayerNorm(config.n_embd, bias=config.bias),
            )
        )

        mtp_config = GPTConfig(
            block_size=config.mtp_horizon + 1,
            vocab_size=config.vocab_size,
            n_layer=1,  # lightweight module
            n_head=min(4, config.n_head),  # match Implementation A
            n_embd=config.n_embd,
            dropout=config.dropout,
            bias=config.bias,
        )
        self.mtp_light = nn.ModuleDict(
            dict(
                relative_attn=RelativeSelfAttention(
                    mtp_config, max_len=config.mtp_horizon + 1
                ),
                teacher_combine=TeacherCombine(config.n_embd),
                norm=LayerNorm(config.n_embd, bias=config.bias),
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
        return_hidden_states: bool = False,  # Whether to return hidden states for CVAE
    ) -> torch.Tensor:
        batch_size, seq_len = batch.size()
        mtp_horizon = self.config.mtp_horizon
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
        tok_emb = self.token_embedding(batch)
        x = tok_emb

        for block in self.transformer.blocks:
            x = block(x, mask=mask, rope=rope)
        x = self.transformer.norm(x)

        # Compute teacher token using teacher_combine
        ntp_token = self.mtp_light.teacher_combine(x, tok_emb)  # (B, T, n_embd)

        # Inference mode: use only the ntp branch
        if targets is None:
            planning_seq = ntp_token.view(
                batch_size * seq_len, 1, -1
            )  # (B*T, 1, n_embd)
            planning_out = self.mtp_light.relative_attn(
                planning_seq
            )  # (B*T, 1, n_embd)
            planning_out = planning_out.view(batch_size, seq_len, -1)
            ntp_final = self.mtp_light.norm(planning_out)
            output = self.lm_head(ntp_final)

        # If targets are given, compute loss
        else:
            # Build teacher and target indices for MTP training
            teacher_idx, target_idx = build_teacher_and_target_idx(
                batch,
                batch_size,
                seq_len,
                mtp_horizon,
                self.config.context_length,
                pos,
                self.config.eos_token_id,
                batch.device,
            )
            ntp_token_flat = ntp_token.view(
                batch_size * seq_len, 1, -1
            )  # (B*T, 1, n_embd)
            teacher_idx_flat = teacher_idx.view(
                batch_size * seq_len, mtp_horizon
            )  # (B*T, D)
            teacher_emb = self.token_embedding(teacher_idx_flat)  # (B*T, D, n_embd)
            x_flat = x.view(batch_size * seq_len, -1)  # (B*T, n_embd)
            x_expanded = x_flat.unsqueeze(1).expand(
                batch_size * seq_len, mtp_horizon, x_flat.size(-1)
            )  # (B*T, D, n_embd)

            # Compute mtp token using teacher_combine
            mtp_token = self.mtp_light.teacher_combine(
                x_expanded, teacher_emb
            )  # (B*T, D, n_embd)

            # Concatenate ntp and mtp tokens to form a planning sequence
            planning_seq = torch.cat(
                [ntp_token_flat, mtp_token], dim=1
            )  # (B*T, D+1, n_embd)

            # Run relative self-attention
            planning_out = self.mtp_light.relative_attn(
                planning_seq
            )  # (B*T, D+1, n_embd)
            planning_out = planning_out.view(
                batch_size, seq_len, mtp_horizon + 1, -1
            )  # (B, T, D+1, n_embd)

            # Split outputs and for mtp branch add skip connection for x
            planning_ntp = planning_out[:, :, 0, :]  # (B, T, n_embd)
            # residual connection for mtp hidden states
            planning_mtp = planning_out[:, :, 1:, :] + x.view(
                batch_size, seq_len, 1, -1
            )  # (B, T, D, n_embd)

            ntp_final = self.mtp_light.norm(planning_ntp)
            mtp_final = self.mtp_light.norm(planning_mtp)

            # Compute losses.
            targets_ntp = targets
            if self.config.context_length > 0:
                pos_ids = self.model.create_position_indices(targets)
                # Create context mask: mask positions less than context_length
                # Sequence:    [ A,  B,  C, EOS, D, E, F, G, H, EOS, I, J]
                # pos:         [ 1,  2,  3,  0,  1, 2, 3, 4, 5,  0,  1, 2]
                context_length_mask = (pos_ids <= self.config.context_length + 1) & (
                    pos != 0
                )
                targets_ntp = targets_ntp.masked_fill(context_length_mask, -100)

            if targets.shape[-1] != batch.shape[-1]:
                ntp_final = ntp_final[:, :-1]

            loss_ntp = self.cross_entropy_loss(
                input=ntp_final,
                last_layer=self.lm_head,
                targets=targets_ntp,
            )

            loss_mtp_individual = []
            for i in range(mtp_horizon):
                loss_i = self.cross_entropy_loss(
                    input=mtp_final[:, :, i, :],
                    last_layer=self.lm_head,
                    targets=target_idx[:, :, i],
                )
                loss_mtp_individual.append(loss_i)

            loss_mtp_individual = torch.stack(loss_mtp_individual)
            loss = loss_ntp + self.config.mtp_lambda * loss_mtp_individual.mean()

            output = (loss, loss_ntp, loss_mtp_individual)

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


class JTP(ModelBase, SpeculativeModel):
    def __init__(
        self,
        config: MTPJTPConfig,
    ):
        super().__init__()

        self.config = config
        self.model = JTPTransformer(config)

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
        """
        Compute loss on a given batch of data. Optionally run backward pass.
        For gradient accumulation, call this function multiple times, iterating over each sub-batch.
        Loss will be divided by loss_div before backpropagation.
        Returns detached loss as a tensor.
        """
        self._assert_fabric_is_setup()

        # use full input to create teacher and target indices
        inputs = batch

        # If using FSDP, we ignore no_sync and always sync gradients
        # This allows us to re-shard each layer of the model after backward pass
        is_fsdp = isinstance(self.model.module, FSDPModule)
        with self.fabric.no_backward_sync(self.model, no_sync and not is_fsdp):
            (loss, loss_ntp, loss_mtp_individual), hidden_states = self.model(
                inputs, targets=targets, return_hidden_states=True
            )
            loss = loss / loss_div
            loss_ntp = loss_ntp / loss_div
            loss_mtp_individual = loss_mtp_individual / loss_div

            # Backward pass
            if backpropagate:
                self.fabric.backward(loss)

        result = {
            "loss": loss.detach(),
            "next_token_loss": loss_ntp.detach(),
        }
        result.update(
            {
                f"loss_token_{i+1}": loss_mtp_individual[i].detach()
                for i in range(self.config.mtp_horizon)
            }
        )

        if self.config.compute_hidden_state_rank:
            hidden_states_det = hidden_states.detach()
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
        logits, hidden_states = core(seq, return_hidden_states=True)
        x_t = hidden_states[:, -1, :]
        drafted: List[torch.Tensor] = []
        q_probs_steps: List[torch.Tensor] = []

        q1 = normalize_logits(
            logits[:, -1, :], temperature=temperature, top_k=top_k, top_p=top_p
        )
        tok1 = sample_from_probs(q1)
        drafted.append(tok1)
        q_probs_steps.append(q1)

        if steps_to_propose == 1:
            return tok1, q_probs_steps

        last_tok = seq[:, -1]
        last_tok_emb = core.token_embedding(last_tok)
        ntp_token = core.mtp_light.teacher_combine(x_t, last_tok_emb)
        drafted_emb = core.token_embedding(tok1)

        for _ in range(2, steps_to_propose + 1):
            steps = drafted_emb.size(1)
            x_expanded = x_t.unsqueeze(1).expand(-1, steps, -1)
            mtp_tokens = core.mtp_light.teacher_combine(x_expanded, drafted_emb)
            planning_seq = torch.cat([ntp_token.unsqueeze(1), mtp_tokens], dim=1)
            planning_out = core.mtp_light.relative_attn(planning_seq)
            planning_mtp = planning_out[:, 1:, :] + x_t.unsqueeze(1)
            mtp_final = core.mtp_light.norm(planning_mtp)
            mtp_logits = core.lm_head(mtp_final)
            qk = normalize_logits(
                mtp_logits[:, -1, :], temperature=temperature, top_k=top_k, top_p=top_p
            )
            tokk = sample_from_probs(qk)
            drafted.append(tokk)
            q_probs_steps.append(qk)
            drafted_emb = torch.cat((drafted_emb, core.token_embedding(tokk)), dim=1)

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
            logits = self.model(idx_cond)
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
        logits = self.model(batch)

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
