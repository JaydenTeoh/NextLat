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
from typing import Any, Dict, Optional, Tuple

from models.model_base import (
    ModelBase,
    DocumentRelativePositions,
    FusedCrossEntropyLoss,
    LayerNorm,
    RotaryPositionEmbedding,
    SwiGLU,
)
from models.model_gpt import Block


# refer to the BST paper to understand the parameters
# https://arxiv.org/abs/2410.23506
@dataclass
class BSTConfig:
    block_size: int = 1024
    vocab_size: int = -1
    eos_token_id: int = -1
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    bias: bool = False
    context_length: int = 0  # default is zero
    # True: use fused Liger kernels. False: use regular PyTorch functions.
    use_fused: bool = False
    # BST pairing options
    bst_pair_minimum_gap: int = 1
    bst_pair_maximum_gap: int = -1
    bst_pair_subsample_rate: float = 1.0
    bst_single_gap_prediction_mode: str = "eos"


class TextHead(FusedCrossEntropyLoss, nn.Module):
    def __init__(self, config: BSTConfig):
        super().__init__()
        self.config = config
        input_dim = config.n_embd * 2  # forward and backward
        hidden_dim = 8 * input_dim / 3
        hidden_dim = 128 * round(hidden_dim / 128)

        # MLP to combine forward and backward embeddings
        # Input to MLP (output from transformer encoders) is already normalized
        self.mlp = nn.Sequential(
            SwiGLU(input_dim, hidden_dim, bias=config.bias),
            nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, input_dim, bias=config.bias),
        )
        self.norm = LayerNorm(input_dim, bias=config.bias)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        forward_embedding: torch.Tensor,
        backward_embedding: torch.Tensor,
        targets_next: Optional[torch.Tensor] = None,
        targets_prev: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Shape of each embedding is (batch_size, n_embd)
        # Input to hidden layers has shape (batch_size, n_embd * 2)
        x = torch.cat([forward_embedding, backward_embedding], dim=-1)

        # Residual connection
        x = x + self.mlp(x)
        x = self.norm(x)

        # Split the output into two parts for next and previous tokens
        # Shape of each part is (batch_size, n_embd)
        x_next, x_prev = x.chunk(2, dim=-1)

        # If no targets are given, return logits
        if targets_next is None and targets_prev is None:
            # Apply last layers
            logits_next = self.lm_head(x_next)
            logits_prev = self.lm_head(x_prev)

            # Output shape is (batch_size, 2, vocab_size)
            logits = torch.stack([logits_next, logits_prev], dim=1)
            return logits

        # If targets are given, use fused kernel to compute loss
        else:
            loss_next, loss_prev = None, None

            if targets_next is not None:
                loss_next = self.cross_entropy_loss(
                    input=x_next,
                    last_layer=self.lm_head,
                    targets=targets_next,
                )
            if targets_prev is not None:
                loss_prev = self.cross_entropy_loss(
                    input=x_prev,
                    last_layer=self.lm_head,
                    targets=targets_prev,
                )

            return loss_next, loss_prev

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


class TransformerEncoder(DocumentRelativePositions, nn.Module):
    def __init__(self, config: BSTConfig):
        super().__init__()
        self.config = config

        # shared for forward and backward encoders
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)

        # add 1 extra position embedding for implicit EOS at start
        n_pos_embd = config.block_size + 1
        self.rotary_embedding = RotaryPositionEmbedding(
            max_seq_len=n_pos_embd,
            head_dim=config.n_embd // config.n_head,
        )

        self.transformer_f = nn.ModuleDict(
            dict(
                blocks=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                norm=LayerNorm(config.n_embd, bias=config.bias),
            )
        )

        self.transformer_b = nn.ModuleDict(
            dict(
                blocks=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                norm=LayerNorm(config.n_embd, bias=config.bias),
            )
        )

        # init all weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_num_params(self, non_embedding=True) -> int:
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the token embeddings get subtracted.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.token_embedding.weight.numel()
        return n_params

    @torch.no_grad()
    def create_attention_masks(
        self, batch: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create forward and backward attention masks for a batch of token sequences
        Mask will avoid attending to tokens in different documents
        """
        # find positions of eos tokens
        # shape is (batch_size, seq_len)
        eos_positions = batch == self.config.eos_token_id

        # create indices of packed documents in the token sequence
        # each token can attend up to and including previous EOS, but not next EOS
        # shape is (batch_size, seq_len)
        document_id = torch.cumsum(eos_positions, dim=1)

        # create mask of tokens within the same document
        # shape is (batch_size, seq_len, seq_len)
        mask_f = document_id.unsqueeze(1) == document_id.unsqueeze(2)
        # forward encoder uses causal mask
        mask_f = torch.tril(mask_f)

        # for backward mask, shift document indices by 1
        # each token can attend up to and including next EOS, but not previous EOS
        document_id = document_id.roll(shifts=1, dims=1)
        document_id[:, 0] = 0  # first token is always start of first document

        # create mask of tokens within the same document
        # shape is (batch_size, seq_len, seq_len)
        mask_b = document_id.unsqueeze(1) == document_id.unsqueeze(2)
        # backward encoder uses reverse causal mask
        mask_b = torch.triu(mask_b)

        return mask_f, mask_b

    @torch.no_grad()
    def create_position_indices(
        self, batch: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create forward and backward position indices for a batch of token sequences.
        Position indices are relative to each document, and a sequence can have multiple packed documents.
        Forward indices are from left to right, and backward indices are from right to left.
        EOS tokens always have position 0.

        For example
            Sequence: [A, B, C, EOS, D, E, F, G, H, EOS, I, J]
            Forward:  [1, 2, 3,  0,  1, 2, 3, 4, 5,  0,  1, 2]
            Backward: [3, 2, 1,  0,  5, 4, 3, 2, 1,  0,  2, 1]
        """
        pos_indices_fwd = super().create_position_indices(batch)

        # Reverse sequences to create backward position indices
        batch = batch.flip(dims=(1,))
        pos_indices_bwd = super().create_position_indices(batch)
        pos_indices_bwd = pos_indices_bwd.flip(dims=(1,))

        return pos_indices_fwd, pos_indices_bwd

    def forward(
        self,
        batch: torch.Tensor,
        compute_forward: bool = True,
        compute_backward: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Pass a batch of token sequences through the forward and backward encoders
        Returns the output of each encoder
        """
        # batch has shape (batch_size, seq_len)
        mask_f, mask_b = self.create_attention_masks(batch)
        pos_f, pos_b = self.create_position_indices(batch)

        if compute_forward:
            fwd = self.raw_f_enc(batch, pos=pos_f, mask=mask_f)
        else:
            fwd = None

        if compute_backward:
            bwd = self.raw_b_enc(batch, pos=pos_b, mask=mask_b)
        else:
            bwd = None

        # Each output has shape (batch_size, seq_len, n_embd)
        return fwd, bwd

    def raw_f_enc(self, batch, pos=None, mask=None) -> torch.Tensor:
        # batch has shape (batch_size, seq_len)
        _, seq_len = batch.shape
        device = batch.device

        if pos is None:
            # default token positions are just 0, 1, 2, ..., t-1
            pos = torch.arange(0, seq_len, dtype=torch.long, device=device)
        rope = self.rotary_embedding(pos)

        # token embeddings of shape (b, t, n_embd)
        fwd = self.token_embedding(batch)

        for block in self.transformer_f.blocks:
            fwd = block(fwd, mask=mask, rope=rope)
        fwd = self.transformer_f.norm(fwd)

        return fwd

    def raw_b_enc(self, batch, pos=None, mask=None) -> torch.Tensor:
        # batch has shape (batch_size, seq_len)
        _, seq_len = batch.shape
        device = batch.device

        if pos is None:
            # default token positions are just 0, 1, 2, ..., t-1
            pos = torch.arange(0, seq_len, dtype=torch.long, device=device)
        rope = self.rotary_embedding(pos)

        # backward encoder uses reverse attention mask
        # if mask is None, we must manually create it
        if mask is None:
            mask = torch.triu(
                torch.ones(seq_len, seq_len, dtype=torch.bool, device=device)
            )

        # token embeddings of shape (b, t, n_embd)
        bwd = self.token_embedding(batch)

        for block in self.transformer_b.blocks:
            bwd = block(bwd, mask=mask, rope=rope)
        bwd = self.transformer_b.norm(bwd)

        return bwd

    def compile(self):
        # Don't compile forward() because it gives an error with gradient accumulation
        # Instead compile functions raw_f_enc() and raw_b_enc() that are called in forward()
        self.raw_f_enc = torch.compile(self.raw_f_enc)
        self.raw_b_enc = torch.compile(self.raw_b_enc)
        # Also compile helper functions called by forward()
        self.create_attention_masks = torch.compile(self.create_attention_masks)
        self.create_position_indices = torch.compile(self.create_position_indices)


class BST(ModelBase):
    def __init__(
        self,
        config: BSTConfig,
    ):
        super().__init__()
        assert config.vocab_size is not None and config.vocab_size > 0
        assert config.block_size is not None and config.block_size > 0
        assert config.eos_token_id is not None and config.eos_token_id >= 0
        assert 0 <= config.bst_pair_subsample_rate <= 1
        self.config = config

        self.encoder = TransformerEncoder(config)
        self.text_head = TextHead(config)

        # To be filled in by setup_fabric()
        self.pairing_stream: torch.cuda.Stream = None

        # Statistics for subsampling pairs
        self._avg_valid_pairs = None
        self._avg_sampled_pairs = None
        self._n_iters_averaged = 0

    def get_num_params(self, non_embedding: bool = True) -> int:
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        return (
            self.encoder.get_num_params(non_embedding=non_embedding)
            + self.text_head.get_num_params()
        )

    def _create_pair_indices(
        self,
        document_len: int,
        start_index: int = 0,
        min_gap: int = 1,
        max_gap=-1,
        device=None,
    ) -> torch.Tensor:
        """
        Generate pairs (t, t+k) for the forward and backward encoders
        document_length is the length of the document
        start_index is the starting index for this document's pairs
        This allows us to handle packed sequences with multiple documents
        """
        # Set maximum gap to document length if not defined
        if max_gap <= 0:
            max_gap = document_len
        assert max_gap >= min_gap, "max_gap must be greater than min_gap"

        start = torch.arange(start_index, document_len, dtype=torch.long, device=device)
        offset = torch.arange(min_gap, max_gap, dtype=torch.long, device=device)
        combinations = torch.cartesian_prod(start, offset)

        # convert (start, offset) to (start, end)
        combinations[:, 1] = combinations[:, 0] + combinations[:, 1]

        # make sure no pair goes over limit
        fb_pairs = combinations[combinations[:, 1] < document_len]

        return fb_pairs

    @torch.no_grad()
    def _create_valid_pairs(self, seq: torch.Tensor) -> torch.Tensor:
        """
        Create valid forward-backward pairs for a sequence of tokens.
        Valid pairs have prefix and suffix tokens that are within the same document.

        Batch size must be 1, so idx has shape (seq_len)
        Output has shape (n_pairs, 2), where n_pairs is dependent on the input sequence

        If subsample is between 0 and 1, randomly select pairs with the specified probability.
        Always keep pairs that contain an EOS token, so we train fully for unconditional generation.
        """
        assert seq.ndim == 1
        seq_len = seq.size(0)

        # create indices of all possible token pairs
        # shape is (n_possible_pairs, 2)
        all_pairs = self._create_pair_indices(
            seq_len,
            start_index=self.config.context_length,
            min_gap=self.config.bst_pair_minimum_gap,
            max_gap=self.config.bst_pair_maximum_gap,
            device=seq.device,
        )

        # find positions of EOS tokens
        eos_positions = seq == self.config.eos_token_id

        # create indices of packed documents in the token sequence
        # EOS token is the start of a new document
        # shape is (seq_len)
        document_id_fwd = torch.cumsum(eos_positions, dim=0)

        # shift document indices so that EOS is at the end of the document
        # shape is (seq_len)
        document_id_bwd = document_id_fwd.roll(shifts=1, dims=0)
        document_id_bwd[0] = 0  # first token is always start of first document

        # filter out pairs that cross between different documents
        # shape is (seq_len)
        valid_pairs_mask = (
            document_id_fwd[all_pairs[:, 0]] == document_id_bwd[all_pairs[:, 1]]
        )

        # shape is (n_valid_pairs, 2)
        valid_pairs = all_pairs[valid_pairs_mask]
        n_valid_pairs = valid_pairs.size(0)

        # subsampling can be used to reduce the number of pairs we actually train on
        target_n_pairs = self._get_pair_subsample_target()
        if target_n_pairs is None:
            # keep all valid pairs
            sampled_pairs = valid_pairs
        else:
            # get pairs where at least one of the tokens is EOS
            # shape is (n_valid_pairs)
            eos_pairs_mask = (
                eos_positions[valid_pairs[:, 0]] | eos_positions[valid_pairs[:, 1]]
            )
            n_eos_pairs = eos_pairs_mask.sum()

            # probability chosen so that the expected number of pairs is target_n_pairs
            probability = (target_n_pairs - n_eos_pairs) / (n_valid_pairs - n_eos_pairs)
            # randomly select pairs with the calculated probability
            random_mask = torch.rand(n_valid_pairs, device=seq.device) < probability
            # combine randomly selected pairs with EOS pairs
            sampled_pairs = valid_pairs[eos_pairs_mask | random_mask]

        # if no pairs sampled, return empty tensor of shape (0, 2)
        n_sampled_pairs = sampled_pairs.size(0)
        if n_sampled_pairs == 0:
            sampled_pairs = torch.zeros(0, 2, dtype=torch.long, device=seq.device)

        self._update_pair_stats(
            n_valid_pairs=n_valid_pairs,
            n_sampled_pairs=n_sampled_pairs,
        )

        return sampled_pairs

    # Compile gives error because control flow here depends on state of BST object
    @torch.compiler.disable()
    def _get_pair_subsample_target(self) -> Optional[int]:
        subsample = self.config.bst_pair_subsample_rate
        if subsample >= 1:
            # if subsample is 1, keep all valid pairs
            return None

        # get average number of valid pairs from previous training steps
        if self._avg_valid_pairs is not None:
            target_n_pairs = int(self._avg_valid_pairs * subsample)
        else:
            # no previous training step
            # estimate total pairs by sequence length
            seq_len = self.config.block_size
            total_pairs = seq_len * (seq_len - 1) // 2
            target_n_pairs = int(total_pairs * subsample)

        return target_n_pairs

    def _update_pair_stats(self, n_valid_pairs: int, n_sampled_pairs: int):
        """
        Keep a running average of the number of valid and sampled pairs
        """
        # update number of times this function has been called
        self._n_iters_averaged += 1
        # decay moving average factor
        alpha = 1.0 / self._n_iters_averaged
        alpha = min(max(alpha, 0.001), 0.1)

        if self._avg_valid_pairs is None:
            self._avg_valid_pairs = n_valid_pairs
        else:
            self._avg_valid_pairs = (
                1 - alpha
            ) * self._avg_valid_pairs + alpha * n_valid_pairs

        if self._avg_sampled_pairs is None:
            self._avg_sampled_pairs = n_sampled_pairs
        else:
            self._avg_sampled_pairs = (
                1 - alpha
            ) * self._avg_sampled_pairs + alpha * n_sampled_pairs

    def get_custom_metrics(self) -> Dict[str, Any]:
        """
        Return pair statistics for logging
        """
        return {
            "avg_valid_pairs": self._avg_valid_pairs,
            "avg_sampled_pairs": self._avg_sampled_pairs,
        }

    def compute_loss(
        self,
        batch: torch.Tensor,  # Shape is (batch_size, seq_len)
        targets: torch.Tensor,  # Shape is (batch_size, seq_len-1)
        pair_batch_size: int,  # Number of pairs to process in a single texthead batch
        backpropagate: bool,  # Run backward pass if true, otherwise only compute loss
        no_sync: bool = False,  # If True, don't sync gradients across multiple GPUs
        loss_div: int = 1,  # Loss will be divided by this number
        **kwargs,  # Extra arguments ignored for compatibility with GPT
    ) -> torch.Tensor:
        """
        Compute loss on a given batch of data. Optionally run backward pass.
        For gradient accumulation, call this function multiple times, iterating over each sub-batch.
        Loss will be divided by loss_div before backpropagation.
        Returns loss over all valid pairs, divided by loss_div, and detached.
        """
        self._assert_fabric_is_setup()

        # batch has shape (batch_size, seq_len)
        batch_size, _ = batch.shape

        # If using FSDP, we ignore no_sync and always sync gradients
        # This allows us to re-shard each layer of the model after backward pass
        is_fsdp = isinstance(self.encoder.module, FSDPModule)

        # First pass the batch through the forward and backward encoders
        # Shape is (batch_size, seq_len, n_embd)
        with self.fabric.no_backward_sync(self.encoder, no_sync and not is_fsdp):
            fwd_emb, bwd_emb = self.encoder(batch)

        # Detach the embeddings to avoid computing gradients through them
        # when accumulating gradients through the text head
        fwd_det = fwd_emb.detach()
        bwd_det = bwd_emb.detach()
        fwd_det.requires_grad_()
        bwd_det.requires_grad_()

        if self.pairing_stream is not None:
            self.pairing_stream.wait_stream(torch.cuda.current_stream())

        # Text head always uses batch size 1
        # Iterate over sequences in batch
        batch_loss = torch.zeros(1, device=batch.device)
        batch_loss_next = torch.zeros(1, device=batch.device)
        batch_loss_prev = torch.zeros(1, device=batch.device)
        for batch_idx in range(batch_size):
            # Sequence embeddings have shape (seq_len, n_embd)
            seq_fwd = fwd_det[batch_idx]
            seq_bwd = bwd_det[batch_idx]

            with torch.cuda.stream(self.pairing_stream):
                # Generate valid pairs for this sequence
                seq = batch[batch_idx]
                pair_idx = self._create_valid_pairs(seq)

                # Generate targets to predict
                next_tokens = seq[pair_idx[:, 0] + 1]
                prev_tokens = seq[pair_idx[:, 1] - 1]

                # handle single gap eos mode
                if self.config.bst_single_gap_prediction_mode == "eos":
                    single_gap_idxs = pair_idx[:, 1] - pair_idx[:, 0] == 1
                    next_tokens[single_gap_idxs] = self.config.eos_token_id
                    prev_tokens[single_gap_idxs] = self.config.eos_token_id

            # Wait for the pairing stream to finish
            if self.pairing_stream is not None:
                torch.cuda.current_stream().wait_stream(self.pairing_stream)

            # As long as the rest of the for-loop does not sync with the pairing stream or with CPU,
            # the next pairing stream iteration can start concurrently with text head backward().
            # Note that this is different from no_sync, which controls gradient sync across multiple GPUs.

            # Divide all pairs into batches of size pair_batch_size
            n_pairs = pair_idx.size(0)
            pair_accum_steps = math.ceil(n_pairs / pair_batch_size)
            texthead_loss_div = loss_div * batch_size * pair_accum_steps

            # Accumulate gradients through the text head
            for i in range(pair_accum_steps):
                start = i * pair_batch_size
                end = min((i + 1) * pair_batch_size, n_pairs)

                # Do not sync texthead gradients if global no_sync=True
                # Otherwise only sync gradients on the last iteration
                texthead_no_sync = (
                    batch_idx < batch_size - 1 or i < pair_accum_steps - 1
                )

                loss, loss_next, loss_prev = self._compute_texthead_loss(
                    seq_fwd,
                    seq_bwd,
                    pair_idx[start:end],
                    targets_next=next_tokens[start:end],
                    targets_prev=prev_tokens[start:end],
                    backpropagate=backpropagate,
                    no_sync=(no_sync or texthead_no_sync),
                    loss_div=texthead_loss_div,
                )
                batch_loss += loss
                batch_loss_next += loss_next
                batch_loss_prev += loss_prev

            # Don't deallocate tensors created in pairing_stream until backward() is done
            # Record an event that defers until queued commands in current stream have completed
            if self.pairing_stream is not None:
                pair_idx.record_stream(torch.cuda.current_stream())
                next_tokens.record_stream(torch.cuda.current_stream())
                prev_tokens.record_stream(torch.cuda.current_stream())

        # Backpropagate accumuated gradient through the forward and backward encoders
        if backpropagate:
            combined_emb = torch.cat([fwd_emb, bwd_emb], dim=0)
            fwd_grad = (
                fwd_det.grad if fwd_det.grad is not None else torch.zeros_like(fwd_det)
            )
            bwd_grad = (
                bwd_det.grad if bwd_det.grad is not None else torch.zeros_like(bwd_det)
            )
            combined_grad = torch.cat([fwd_grad, bwd_grad], dim=0)
            with self.fabric.no_backward_sync(self.encoder, no_sync and not is_fsdp):
                self.fabric.backward(combined_emb, gradient=combined_grad)

        return {
            "loss": batch_loss.detach(),
            "next_token_loss": batch_loss_next.detach(),
            "bst_prev_token_loss": batch_loss_prev.detach(),
        }

    def _compute_texthead_loss(
        self,
        fwd_emb: torch.Tensor,  # shape is (seq_len, n_embd)
        bwd_emb: torch.Tensor,  # shape is (seq_len, n_embd)
        pair_idx: torch.Tensor,  # shape is (n_pairs, 2)
        targets_next: torch.Tensor,  # shape is (n_pairs)
        targets_prev: torch.Tensor,  # shape is (n_pairs)
        backpropagate: bool,  # Run backward pass if true, otherwise only compute loss
        no_sync: bool = False,  # If True, don't sync gradients across multiple GPUs
        loss_div=1,  # Total gradient accumulation steps
    ) -> torch.Tensor:
        """
        Compute loss for text head using the given pair indices. Optionally run backward pass.
        Outputs from the encoder (fwd_emb, bwd_emb) should be detached.
        Batch size for forward and backward embeddings must be 1. Batch dimension is omitted.
        All pairs are processed through the text head in a single batch.
        Returns the mean loss over pairs, scaled by loss_div, and detached.
        """
        assert fwd_emb.ndim == 2
        assert bwd_emb.ndim == 2
        assert fwd_emb.size(0) == bwd_emb.size(0)

        # Create a batch across all pairs
        # Shape is (n_pairs, n_embd)
        forward_batch = fwd_emb[pair_idx[:, 0]]
        backward_batch = bwd_emb[pair_idx[:, 1]]

        # Unlike the encoder, we actually use no_sync for FSDP text head
        # This requires keeping the full params and gradients during accumulation
        # Only reshard on the last backward pass when we sync gradients
        # This is necessary so that different GPUs can do different numbers of pairs
        with self.fabric.no_backward_sync(self.text_head, no_sync):
            if isinstance(self.text_head.module, FSDPModule):
                self.text_head.module.set_reshard_after_backward(not no_sync)

            loss_next, loss_prev = self.text_head(
                forward_batch,
                backward_batch,
                targets_next=targets_next,
                targets_prev=targets_prev,
            )

            loss_next = loss_next / loss_div
            loss_prev = loss_prev / loss_div
            loss = (loss_next + loss_prev) / 2.0

            # This backpropagates only through the text head
            if backpropagate:
                self.fabric.backward(loss)

        return loss.detach(), loss_next.detach(), loss_prev.detach()

    def setup_fabric(self, fabric: L.Fabric):
        """
        Setup Lightning Fabric for distributed training
        This wraps the models and optimizer with a FabricModule
        """
        self.fabric = fabric

        if not is_wrapped(self.encoder):
            assert not is_wrapped(self.text_head)
            self.encoder = fabric.setup_module(self.encoder)
            self.text_head = fabric.setup_module(self.text_head)

            # Print model architecture
            fabric.print(self.encoder)
            fabric.print(self.text_head)
            fabric.print(
                f"Total encoder parameters: {self.encoder.get_num_params(non_embedding=False):,}"
            )
            fabric.print(
                f"Total text head parameters: {self.text_head.get_num_params():,}"
            )

            # Fabric tells us if we are running on CUDA
            # Create a concurrent CUDA stream for generating pairs
            if fabric.device.type == "cuda":
                self.pairing_stream = torch.cuda.Stream()

        # self.encoder is wrapped, so setup_fabric() was already called
        else:
            assert is_wrapped(self.text_head)

        self._setup_optimizer_with_fabric(fabric)

    def compile(self):
        """
        Compile the model using torch.compile()
        """
        # Must compile before setup_fabric()
        self._assert_fabric_is_setup(setup=False)
        self.encoder.compile()
        self.text_head.compile()
        self._create_valid_pairs = torch.compile(self._create_valid_pairs)

    def train(self):
        """
        Set the model to training mode
        """
        self.encoder.train()
        self.text_head.train()

    def eval(self):
        """
        Set the model to evaluation mode
        """
        self.encoder.eval()
        self.text_head.eval()

    def _optimizer_modules(self) -> Tuple[nn.Module, ...]:
        return (self.encoder, self.text_head)

    def _get_checkpoint_state(self) -> Dict[str, Any]:
        """
        Get the state of the model for checkpoint save/load
        """
        return {
            "encoder": self.encoder,
            "text_head": self.text_head,
            "optimizer": self.optimizer,
            "training_steps": self.training_steps,
        }

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS"""
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd // cfg.n_head, cfg.block_size
        flops_per_token = 6 * N + 12 * L * H * Q * T
        flops_per_fwdbwd = flops_per_token * T * 2
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0 / dt)  # per second
        flops_promised = 1980e12  # H100 GPU bfloat16 peak flops is 1980 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.inference_mode()
    def generate(self, batch, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices `batch` (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        # batch has shape (batch_size, seq_len)
        batch_size, seq_len = batch.shape

        # backward embedding computed with tensor of EOS tokens
        eos_tensor = torch.tensor(
            [self.config.eos_token_id], device=batch.device
        ).expand(batch_size, 1)
        _, bwd_emb = self.encoder(
            eos_tensor, compute_forward=False, compute_backward=True
        )

        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            batch_crop = (
                batch
                if seq_len <= self.config.block_size
                else batch[:, -self.config.block_size :]
            )

            # forward the model to get the logits for the index in the sequence
            fwd_emb, _ = self.encoder(
                batch_crop, compute_forward=True, compute_backward=False
            )
            # get the embedding at the last token
            fwd_emb = fwd_emb[:, -1:, :]

            # pass the forward and backward embeddings through the text head
            # shape is (batch_size, 2, vocab_size)
            logits_next_prev = self.text_head(fwd_emb, bwd_emb)

            # get the logits for just the next token
            # shape is (batch_size, vocab_size)
            logits = logits_next_prev[:, 0, :]

            # scale by desired temperature
            logits = logits / temperature

            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")

            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)

            # Squeeze out the middle dimension if it exists
            if probs.dim() == 3:
                probs = probs.squeeze(1)  # Remove the middle dimension

            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)

            # append sampled index to the running sequence and continue
            batch = torch.cat((batch, idx_next), dim=1)

        return batch

    @torch.inference_mode()
    def evaluation_preds(
        self,
        batch: torch.Tensor,  # Shape is (batch_size, seq_len)
        prefix_end_index: torch.Tensor,  # Shape is (batch_size)
        suffix_start_index: torch.Tensor,  # Shape is (batch_size)
    ) -> torch.Tensor:
        batch_size, seq_len = batch.size()

        # Get the start and end indices of the generated portion of the sequence
        # Shape is (batch_size)
        gen_start_index = prefix_end_index + 1
        gen_end_index = suffix_start_index

        # First pass the batch through the forward and backward encoders
        # Shape is (batch_size, seq_len, n_embd)
        fwd_emb, bwd_emb = self.encoder(batch)

        # Get embedding for just the EOS token
        eos_tensor = torch.tensor(
            [self.config.eos_token_id], device=batch.device, dtype=batch.dtype
        ).expand(batch_size, 1)
        fwd_eos_emb, bwd_eos_emb = self.encoder(eos_tensor)

        # Create concatenated embeddings EOS...sequence...EOS
        # Shape is (batch_size, seq_len + 2, n_embd)
        fwd_emb = torch.cat([fwd_eos_emb, fwd_emb, fwd_eos_emb], dim=1)
        bwd_emb = torch.cat([bwd_eos_emb, bwd_emb, bwd_eos_emb], dim=1)

        # Increment index because of extra EOS at start
        prefix_end_index = prefix_end_index + 1
        suffix_start_index = suffix_start_index + 1

        # Get embeddings for each sequence's prefix and suffix
        # Shape is (batch_size, n_embd)
        # prefix_emb = fwd_emb[torch.arange(batch_size), prefix_end_index]
        suffix_emb = bwd_emb[torch.arange(batch_size), suffix_start_index]

        fwd_texthead = fwd_emb[:, gen_start_index : gen_end_index + 1]
        bwd_texthead = suffix_emb[0].expand_as(fwd_texthead)

        logits_nextprev = self.text_head(fwd_texthead, bwd_texthead)

        logits_next = logits_nextprev[:, 0, :]
        argmax_next = logits_next.argmax(dim=-1)

        print("true", batch[0])
        print("pred", argmax_next[0])

        return logits_nextprev

    @torch.inference_mode()
    def evaluation_loss(
        self,
        batch: torch.Tensor,  # Shape is (batch_size, seq_len)
        prefix_end_index: torch.Tensor,  # Shape is (batch_size)
        suffix_start_index: torch.Tensor,  # Shape is (batch_size)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute next-token and previous-token prediction loss on the given batch of sequences.

        Prefix and suffix tokens for the prompt are excluded from loss computation.
            prefix_end_index: index of the last token in the prefix
            suffix_start_index: index of the first token in the suffix

        If there is no prefix or suffix for a sequence, set its prefix/suffix index to -1.
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
        gen_end_index[no_suffix] = seq_len - 1

        # First pass the batch through the forward and backward encoders
        # Shape is (batch_size, seq_len, n_embd)
        fwd_emb, bwd_emb = self.encoder(batch)

        # Get embedding for just the EOS token
        eos_tensor = torch.tensor(
            [self.config.eos_token_id], device=batch.device, dtype=batch.dtype
        ).expand(batch_size, 1)
        fwd_eos_emb, bwd_eos_emb = self.encoder(eos_tensor)

        # Create concatenated embeddings EOS...sequence...EOS
        # Shape is (batch_size, seq_len + 2, n_embd)
        fwd_emb = torch.cat([fwd_eos_emb, fwd_emb, fwd_eos_emb], dim=1)
        bwd_emb = torch.cat([bwd_eos_emb, bwd_emb, bwd_eos_emb], dim=1)

        # Increment index because of extra EOS at start
        prefix_end_index = prefix_end_index + 1
        suffix_start_index = suffix_start_index + 1

        # No prefix and no suffix map to the new EOS tokens at start/end
        prefix_end_index[no_prefix] = 0
        suffix_start_index[no_suffix] = seq_len + 1

        # Get embeddings for each sequence's prefix and suffix
        # Shape is (batch_size, n_embd)
        prefix_emb = fwd_emb[torch.arange(batch_size), prefix_end_index]
        suffix_emb = bwd_emb[torch.arange(batch_size), suffix_start_index]

        # Compute loss for each sequence in batch
        # Text head is always batch size 1
        next_token_loss = torch.zeros(batch_size, device=batch.device)
        prev_token_loss = torch.zeros(batch_size, device=batch.device)
        for i in range(batch_size):
            # Get the target tokens for this sequence
            # Indices are inclusive of start and end
            targets = batch[i, gen_start_index[i] : gen_end_index[i] + 1]

            # Loss for forward prediction
            # fwd_emb is shifted by 1 token, so we can directly index to get the previous token
            fwd_texthead = fwd_emb[i, gen_start_index[i] : gen_end_index[i] + 1]
            bwd_texthead = suffix_emb[i].expand_as(fwd_texthead)
            loss_next, _ = self.text_head(
                fwd_texthead, bwd_texthead, targets_next=targets
            )
            next_token_loss[i] = loss_next

            # Loss for backward prediction
            # bwd_emb is shifted by 1 token, so we must add 2 to get the next token
            bwd_texthead = bwd_emb[i, gen_start_index[i] + 2 : gen_end_index[i] + 3]
            fwd_texthead = prefix_emb[i].expand_as(bwd_texthead)
            _, loss_prev = self.text_head(
                fwd_texthead, bwd_texthead, targets_prev=targets
            )
            prev_token_loss[i] = loss_prev

        return next_token_loss, prev_token_loss
