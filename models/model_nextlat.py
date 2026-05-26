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
from models.model_gpt import Block, MLP, CausalSelfAttention


@dataclass
class NextLatConfig:
    block_size: int = 1024
    # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    bias: bool = False
    context_length: int = 0
    eos_token_id: int = -1
    compute_hidden_state_rank: bool = False
    # True: use fused Liger kernels. False: use regular PyTorch functions.
    use_fused: bool = False
    # NextLat params
    lambda_kl: float = 1.0  # lambda_KL in the paper
    lambda_mse: float = 1.0  # lambda_MSE in the paper
    lambda_ce: float = 0.0  # optional CE loss on next-next-token prediction
    mtp_horizon: int = 1  # multi-step prediction horizon (d) in the paper
    proj_factor: float = 1.0  # projection factor in the latent dynamics model MLP


class NextLatDynamicsModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        input_dim = config.n_embd * 2  # hidden states and next token embeddings
        hidden_dim = config.proj_factor * input_dim
        hidden_dim = 128 * round(hidden_dim / 128)

        # MLP to combine forward and backward embeddings
        # Input to MLP (output from transformer encoders) is already normalized
        self.hidden_state_dropout = (
            nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()
        )
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=config.bias),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim, bias=config.bias),
            nn.GELU(),
            nn.Linear(hidden_dim, config.n_embd, bias=config.bias),
        )
        self.norm_x = LayerNorm(input_dim, bias=config.bias)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward(
        self,
        current_states: torch.Tensor,
        next_token_embeds: torch.Tensor,
    ) -> torch.Tensor:
        # Shape of each embedding is (batch_size, n_embd)
        # Input to hidden layers has shape (batch_size, n_embd * 2)
        hidden_states = self.hidden_state_dropout(current_states)
        x = torch.cat([next_token_embeds, hidden_states], dim=-1)
        x = self.norm_x(x)

        # residual connection
        delta = self.mlp(x)
        next_states = delta + current_states

        return next_states

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


class NextLatTransformer(DocumentRelativePositions, FusedCrossEntropyLoss, nn.Module):
    def __init__(self, config: NextLatConfig):
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
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.dynamics_model = NextLatDynamicsModel(config)

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
        token_embeds = x

        for block in self.transformer.blocks:
            x = block(x, mask=mask, rope=rope)
        text_embd = self.transformer.norm(x)

        if return_hidden_states:
            return token_embeds, text_embd

        # If no targets given, return logits
        if targets is None:
            output = self.lm_head(text_embd)

        # If targets are given, compute loss
        else:
            # We pass in all tokens in sequence to get hidden states
            # We need to exclude the last token from the ntp loss
            loss_text_embd = text_embd[:, :-1]
            output = self.cross_entropy_loss(
                input=loss_text_embd,
                last_layer=self.lm_head,
                targets=targets,
            )

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


class NextLat(ModelBase, SpeculativeModel):
    def __init__(
        self,
        config: NextLatConfig,
    ):
        super().__init__()

        self.config = config
        assert config.mtp_horizon >= 1, "mtp_horizon must be at least 1"
        self.model = NextLatTransformer(config)

    def setup_fabric(self, fabric: L.Fabric):
        """
        Setup Lightning Fabric for distributed training
        This wraps the models and optimizer with a FabricModule
        """
        self.fabric = fabric

        if not is_wrapped(self.model):
            self.model = fabric.setup_module(self.model)
            # Allow calling this helper through the Fabric wrapper.
            self.model.mark_forward_method("cross_entropy_loss")
            # Print model architecture
            self.fabric.print(self.model)
            self.fabric.print(f"Total number of parameters: {self.get_num_params():,}")

        self._setup_optimizer_with_fabric(fabric)

    def _categorical_kl_loss(
        self,
        logits_post: torch.Tensor,
        logits_prior: torch.Tensor,
        token_pred_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute KL(logits_post || logits_prior).
        Uses F.kl_div with reduction='none' (because it doesn't support masking),
        then applies mask on per-token KL to get batchmean reduction.
        """
        # posterior: log q (teacher logits)
        log_q = F.log_softmax(logits_post, dim=-1)
        # prior: log p (student logits)
        log_p = F.log_softmax(logits_prior, dim=-1)

        # pointwise KL over vocab, shape: (B, T, V)
        kl_pointwise = F.kl_div(log_p, log_q, log_target=True, reduction="none")
        # per-token KL, shape: (B, T)
        kl_per_token = kl_pointwise.sum(dim=-1)
        mask = token_pred_mask.to(dtype=kl_per_token.dtype)
        # batchmean reduction over valid positions
        # mask is (B, T) with 1 for valid positions, 0 for invalid positions
        # we reduce the loss over the valid positions
        return (kl_per_token * mask).sum() / mask.sum().clamp_min(1.0)

    def _nextlat_loss_function(
        self,
        pred_h_t_next: torch.Tensor,
        h_t_next: torch.Tensor,
        target_tokens: torch.Tensor,
        teacher_logits: torch.Tensor,
        token_pred_mask: torch.Tensor,
        mse_mask: torch.Tensor,
    ):
        """
        Compute NextLat loss components:
          1) Hidden-state MSE:  ||pred_h_t_next - h_t_next||^2
          2) Next-next-token KL:  KL(teacher_logits || pred_tokens)
          3) Next-next-token CE:  CE(target_tokens || pred_tokens)
        """
        # ------------------------------------------------------------
        # Next-Latent MSE loss
        # ------------------------------------------------------------
        # stop gradient on h_t_next
        mse_elem = F.smooth_l1_loss(pred_h_t_next, h_t_next.detach(), reduction="none")
        w = mse_mask.unsqueeze(-1).to(dtype=mse_elem.dtype)
        # Same as reduction="mean" over masked (B, T, n_embd) elements.
        # i.e., divide over B*T*n_embd elements.
        denom = w.expand_as(mse_elem).sum().clamp_min(1.0)
        MSE = (mse_elem * w).sum() / denom

        # ------------------------------------------------------------
        # Next-Latent KL/CE losses
        # ------------------------------------------------------------
        # Align prediction length with target length:
        # - Standard shifted layout: pred_h_t_next is one step longer than target_tokens
        #   (the last predicted hidden state has no corresponding next-token target),
        #   so we drop trailing positions.
        # - A5 layout: pred_h_t_next and target_tokens already share the same length,
        #   so no slicing is needed.
        pred_len = pred_h_t_next.shape[1]
        tgt_len = target_tokens.shape[1]
        if pred_len > tgt_len:
            pred_token_inputs = pred_h_t_next[:, :tgt_len]
        else:
            pred_token_inputs = pred_h_t_next
        # Keep NextLat CE/KL from updating lm_head by using detached head weights.
        # Have to materialize student logits to compute KL loss
        lm_head_weight = self.model.lm_head.weight.detach()
        pred_tokens = F.linear(pred_token_inputs, lm_head_weight)
        targets_masked = target_tokens.masked_fill(~token_pred_mask, -100)
        token_loss = F.cross_entropy(
            pred_tokens.reshape(-1, pred_tokens.size(-1)),
            targets_masked.reshape(-1),
            ignore_index=-100,
            reduction="mean",
        )
        # Detach teacher logits to avoid updating teacher computation graph.
        # teacher_logits may also be longer than pred_tokens in the standard layout;
        # trim to the same length so KL is well-defined.
        teacher_logits_aligned = teacher_logits[:, : pred_tokens.shape[1]]
        kl_loss = self._categorical_kl_loss(
            teacher_logits_aligned.detach(), pred_tokens, token_pred_mask
        )

        return MSE, token_loss, kl_loss

    def _nextlat_compute_losses(
        self,
        hidden_states_det: torch.Tensor,
        next_token_embeds_det: torch.Tensor,
        targets: torch.Tensor,
        nextlat_token_pred_mask: torch.Tensor,
        curr_token_is_eos: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        We separate _nextlat_compute_losses into a separate function so that
        tensor computations can be compiled by torch.compile.
        """
        # Create references for gradient accumulation later
        next_states = hidden_states_det  # Shape: (batch_size, seq_len, n_embd)
        pred_next_states = hidden_states_det
        next_tokens = next_token_embeds_det

        # ------------------------------------------------------------
        # Standard next-token prediction loss
        # ------------------------------------------------------------
        if targets.shape[1] != hidden_states_det.shape[1]:
            # Standard shifted next-token layout: targets has length seq_len-1
            text_embd = hidden_states_det[:, :-1]
        else:
            # A5 layout: targets is a separate length-T block aligned with all T hidden states
            text_embd = hidden_states_det
        # We need explicit logits for KL(teacher || student)
        teacher_logits = self.model.lm_head(text_embd)
        ntp_loss = F.cross_entropy(
            teacher_logits.reshape(-1, teacher_logits.size(-1)),
            targets.reshape(-1),
            reduction="mean",
        )

        # ------------------------------------------------------------
        # Next-Latent prediction loss
        # ------------------------------------------------------------
        total_mse_loss = torch.zeros(1, device=hidden_states_det.device)
        total_nextlat_token_loss = torch.zeros(
            self.config.mtp_horizon, device=hidden_states_det.device
        )
        total_kl_loss = torch.zeros(1, device=hidden_states_det.device)
        # We can do multi-step prediction in a recursive manner.
        # P(\hat{h}_{t+1} | h_t, x_{t+1})
        # P(\hat{h}_{t+2} | \hat{h}_{t+1}, x_{t+2})
        # ...
        # P(\hat{h}_{t+k} | \hat{h}_{t+k-1}, x_{t+k})
        for i in range(self.config.mtp_horizon):
            # We shift current hidden states (pred_next_states) by 1 to the left.
            # h_1, h_2, ..., h_T -> h_1, ..., h_{T-1}
            # Then, we shift next token embeddings, target tokens, target hidden states, and teacher logits by 1 to the right.
            # x_1, x_2, ..., x_T -> x_2, ..., x_T
            pred_next_states = pred_next_states[:, :-1]
            next_tokens = next_tokens[:, 1:]
            next_states = next_states[:, 1:]
            targets = targets[:, 1:]
            teacher_logits = teacher_logits[:, 1:]
            nextlat_token_pred_mask = nextlat_token_pred_mask[:, 1:]

            # We now predict the next-hidden state using current hidden states and next token embeddings.
            # P(\hat{h}_{t+1} | h_t, x_{t+1})
            # h_1, h_2, ..., h_{T-1} -> h_2, h_3, ..., h_T
            pred_next_states = self.model.dynamics_model(pred_next_states, next_tokens)

            # skip MSE on h_{t+1} corresponding to EOS token because it crosses document boundaries
            # EOS token is first hidden state in the next document, see create_attention_mask()
            # also, predicting the hidden state of the EOS token is not needed for belief state convergence
            mse_mask = ~curr_token_is_eos[:, i + 1 :]

            # Compute NextLat loss
            mse_loss, nextlat_token_loss, kl_loss = self._nextlat_loss_function(
                pred_next_states,
                next_states,
                targets,
                teacher_logits,
                token_pred_mask=nextlat_token_pred_mask,
                mse_mask=mse_mask,
            )

            total_mse_loss += mse_loss
            total_kl_loss += kl_loss
            total_nextlat_token_loss[i] = nextlat_token_loss

        return ntp_loss, total_mse_loss, total_kl_loss, total_nextlat_token_loss

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

        # Predict next token for all tokens before the last token
        batch_size, seq_len = batch.shape
        inputs = batch

        # Create mask for positions where next token, i.e., x_{t+1}, is EOS
        # These positions should be excluded from NextLat token prediction loss because
        # next next token prediction, i.e, (h_t, x_{t+1}) -> h_{t+1} -> x_{t+2}, crosses documents.
        curr_token_is_eos = targets == self.config.eos_token_id
        nextlat_token_pred_mask = ~curr_token_is_eos
        if targets.shape[1] != inputs.shape[1]:
            nextlat_token_pred_mask = nextlat_token_pred_mask[:, :-1]

        if self.config.context_length > 0:
            # Create context mask: mask positions less than context_length
            # Sequence:    [ A,  B,  C, EOS, D, E, F, G, H, EOS, I, J]
            # pos_ids:     [ 1,  2,  3,  0,  1, 2, 3, 4, 5,  0,  1, 2]
            pos_ids = self.model.create_position_indices(targets)
            context_length_mask = (pos_ids <= self.config.context_length + 1) & (
                pos_ids != 0
            )
            targets = targets.masked_fill(context_length_mask, -100)

            nextlat_token_pred_mask = nextlat_token_pred_mask & ~context_length_mask
            assert nextlat_token_pred_mask.shape == (batch_size, seq_len - 1)

        # If using FSDP, we ignore no_sync and always sync gradients
        # This allows us to re-shard each layer of the model after backward pass
        is_fsdp = isinstance(self.model.module, FSDPModule)
        with self.fabric.no_backward_sync(self.model, no_sync and not is_fsdp):
            next_token_embeds, hidden_states = self.model(
                inputs, return_hidden_states=True
            )

            # Detach hidden states and next token embeddings to accumulate gradients from the losses involving
            # next-hidden states and next-token predictions.
            hidden_states_det = hidden_states.detach()
            hidden_states_det.requires_grad_()
            next_token_embeds_det = next_token_embeds.detach()
            next_token_embeds_det.requires_grad_()

            # Loss runs outside the wrapped module forward; use Fabric autocast so NextLat/CE/KL/etc.
            # follow the same mixed-precision (if enabled) policy as the rest of training.
            with self.fabric.autocast():
                ntp_loss, total_mse_loss, total_kl_loss, total_nextlat_token_loss = (
                    self._nextlat_compute_losses(
                        hidden_states_det,
                        next_token_embeds_det,
                        targets,
                        nextlat_token_pred_mask,
                        curr_token_is_eos,
                    )
                )

                # We average the losses over the multi-step prediction interval.
                total_mse_loss = total_mse_loss / self.config.mtp_horizon
                total_kl_loss = total_kl_loss / self.config.mtp_horizon
                mean_nextlat_token_loss = total_nextlat_token_loss.mean()
                # Usually, lambda_ce is set to 0; only for logging purposes
                # If lambda_ce is non-zero, set lambda_kl to 0 to turn off KL loss
                nextlat_loss = (
                    self.config.lambda_mse * total_mse_loss
                    + self.config.lambda_kl * total_kl_loss
                    + self.config.lambda_ce * mean_nextlat_token_loss
                )

                loss = ntp_loss + nextlat_loss
                loss = loss / loss_div
                ntp_loss = ntp_loss / loss_div
                mse_loss = total_mse_loss / loss_div
                nextlat_token_loss = total_nextlat_token_loss / loss_div
                kl_loss = total_kl_loss / loss_div

            # Backward pass
            # We accumulate gradients from the losses involving next-latent and next-token predictions,
            # then we backpropagate the accumulated gradients through the transformer trunk all at once.
            if backpropagate:
                self.fabric.backward(loss)
                combined_emb = torch.cat([hidden_states, next_token_embeds], dim=0)
                # Get the gradients of the hidden states and next token embeddings.
                hidden_states_grad = (
                    hidden_states_det.grad
                    if hidden_states_det.grad is not None
                    else torch.zeros_like(hidden_states_det)
                )
                next_token_embeds_grad = (
                    next_token_embeds_det.grad
                    if next_token_embeds_det.grad is not None
                    else torch.zeros_like(next_token_embeds_det)
                )
                # Combine the gradients of the hidden states and next token embeddings.
                combined_grad = torch.cat(
                    [hidden_states_grad, next_token_embeds_grad], dim=0
                )
                # Backpropagate the gradients through the transformer trunk.
                self.fabric.backward(combined_emb, gradient=combined_grad)

        result = {
            "loss": loss.detach(),
            "next_token_loss": ntp_loss.detach(),
            "mse_loss": mse_loss.detach(),
            "kl_loss": kl_loss.detach(),
        }
        result.update(
            {
                f"loss_token_{i+1}": nextlat_token_loss[i].detach()
                for i in range(self.config.mtp_horizon)
            }
        )

        # We measure the rank of the hidden states to check if the hidden states are collapsing.
        if self.config.compute_hidden_state_rank:
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
        # nextlat loss functions involve additional tensor computations
        self._nextlat_compute_losses = torch.compile(self._nextlat_compute_losses)

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

    @classmethod
    def from_pretrained(cls, model_type: str, override_args: Optional[dict] = None):
        assert model_type in {"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"}
        override_args = override_args or {}  # default to empty dict
        # only dropout can be overridden see more notes below
        assert all(k == "dropout" for k in override_args)
        from transformers import GPT2LMHeadModel

        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            "gpt2": dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            "gpt2-medium": dict(n_layer=24, n_head=16, n_embd=1024),  # 350M params
            "gpt2-large": dict(n_layer=36, n_head=20, n_embd=1280),  # 774M params
            "gpt2-xl": dict(n_layer=48, n_head=25, n_embd=1600),  # 1558M params
        }[model_type]
        print("forcing vocab_size=50257, block_size=1024, bias=True")
        config_args["vocab_size"] = 50257  # always 50257 for GPT model checkpoints
        config_args["block_size"] = 1024  # always 1024 for GPT model checkpoints
        config_args["bias"] = True  # always True for GPT model checkpoints
        # we can override the dropout rate, if desired
        if "dropout" in override_args:
            print(f"overriding dropout rate to {override_args['dropout']}")
            config_args["dropout"] = override_args["dropout"]
        # create a from-scratch initialized minGPT model
        config = NextLatConfig(**config_args)
        gpt = NextLat(config)
        sd = gpt.model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [
            k for k in sd_keys if not k.endswith(".attn.bias")
        ]  # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [
            k for k in sd_keys_hf if not k.endswith(".attn.masked_bias")
        ]  # ignore these, just a buffer
        sd_keys_hf = [
            k for k in sd_keys_hf if not k.endswith(".attn.bias")
        ]  # same, just the mask (buffer)
        transposed = [
            "attn.c_attn.weight",
            "attn.c_proj.weight",
            "mlp.c_fc.weight",
            "mlp.c_proj.weight",
        ]
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(
            sd_keys
        ), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.inference_mode():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.inference_mode():
                    sd[k].copy_(sd_hf[k])

        return gpt

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
        _, hidden_states = core(seq, return_hidden_states=True)
        state = hidden_states[:, -1, :]
        drafted: List[torch.Tensor] = []
        q_probs_steps: List[torch.Tensor] = []

        for _ in range(steps_to_propose):
            logits = core.lm_head(state)
            q_probs = normalize_logits(
                logits, temperature=temperature, top_k=top_k, top_p=top_p
            )
            tok = sample_from_probs(q_probs)
            drafted.append(tok)
            q_probs_steps.append(q_probs)
            next_token_emb = core.token_embedding(tok).squeeze(1)
            state = core.dynamics_model(state, next_token_emb)

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
