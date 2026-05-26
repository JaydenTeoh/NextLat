import copy
import itertools
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from dataclasses import dataclass, field
from lightning.fabric import is_wrapped
from torch.distributed.fsdp import FSDPModule
from typing import Any, Dict, Optional, Tuple, List, Union

from models.model_base import (
    ModelBase,
    SwiGLU,
)
from models.model_gpt import GPT
from models.model_bst import BST
from models.model_nextlat import NextLat
from models.model_mtp_jtp import JTP


@dataclass
class ProbeConfig:
    # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    bias: bool = False
    vocab_size: int = 50304
    n_embd: int = 768
    dropout: float = 0.0
    # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    eos_token_id: int = -1
    # True: use fused Liger kernels. False: use regular PyTorch functions.
    use_fused: bool = False
    # Probe parameters
    probe_depth: int = 10


class Probe(nn.Module):
    def __init__(self, config: ProbeConfig):
        super().__init__()
        self.config = config
        self.linear_probes = nn.ModuleList(
            [
                nn.Linear(config.n_embd, config.vocab_size, bias=False)
                for _ in range(config.probe_depth)
            ]
        )
        dec_input_dim = config.n_embd * 2
        dec_output_dim = config.n_embd

        hidden_dim = 8 * dec_input_dim / 3
        hidden_dim = 128 * round(hidden_dim / 128)
        self.decoder = nn.Sequential(
            SwiGLU(dec_input_dim, hidden_dim, bias=config.bias),
            nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, dec_output_dim, bias=config.bias),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        targets: torch.Tensor,
        token_embeddings: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # one-hot encode targets
        one_hot_targets = F.one_hot(targets, num_classes=self.config.vocab_size).float()
        token_embs = one_hot_targets @ token_embeddings

        # pass through decoder
        predict_h_t_next = self.decoder(
            torch.cat([hidden_states[:, :-1], token_embs], dim=-1)
        )

        # pass through linear probes
        logits = torch.stack(
            [probe(hidden_states) for probe in self.linear_probes], dim=-1
        )

        return logits, predict_h_t_next


class ProbeModule(ModelBase):
    def __init__(
        self,
        config: ProbeConfig,
        frozen_model: Union[NextLat, GPT, BST, JTP],
    ):
        super().__init__()

        self.config = config
        self.frozen_model = frozen_model
        # Ensure all frozen model parameters have requires_grad=False
        if isinstance(self.frozen_model, BST):
            for param in self.frozen_model.encoder.parameters():
                param.requires_grad = False
            for param in self.frozen_model.text_head.parameters():
                param.requires_grad = False
        else:
            for param in self.frozen_model.model.parameters():
                param.requires_grad = False

        self.probe = Probe(config)

    def setup_fabric(self, fabric: L.Fabric):
        """
        Setup Lightning Fabric for distributed training
        This wraps the models and optimizer with a FabricModule
        """
        self.fabric = fabric

        if not is_wrapped(self.probe):
            self.probe = fabric.setup_module(self.probe)
            # Print model architecture
            self.fabric.print(self.probe)

        self._setup_optimizer_with_fabric(fabric)

    def compute_loss(
        self,
        batch: torch.Tensor,  # Shape is (batch_size, seq_len)
        targets: torch.Tensor,  # Shape is (batch_size, seq_len-1)
        backpropagate: bool,  # Run backward pass if true, otherwise only compute loss
        loss_div: int = 1,  # Loss will be divided by this number
        eval_mode: bool = False,  # If True, use eval mode for CVAE
        **kwargs,  # Extra arguments ignored for compatibility with BST
    ) -> torch.Tensor:
        """
        Compute combined loss on a given batch of data:
        1. Next token prediction loss (standard GPT)
        2. Hidden state prediction loss (CVAE) using only the last layer's hidden states
        3. Token prediction loss from CVAE's predicted hidden states using transformer's lm_head
           to predict the next next token, masked for positions where next token is EOS
        """
        # Assert that frozen model parameters don't require gradients
        if isinstance(self.frozen_model, BST):
            frozen_params_requiring_grad = [
                name
                for name, param in self.frozen_model.encoder.named_parameters()
                if param.requires_grad
            ]
        else:
            frozen_params_requiring_grad = [
                name
                for name, param in self.frozen_model.model.named_parameters()
                if param.requires_grad
            ]
        assert (
            len(frozen_params_requiring_grad) == 0
        ), f"Frozen model has {len(frozen_params_requiring_grad)} parameters requiring gradients: {frozen_params_requiring_grad[:5]}"

        # Predict next token for all tokens before the last token
        inputs = batch  # Shape: (batch_size, seq_len)
        with torch.no_grad():
            if isinstance(self.frozen_model, BST):
                hidden_states, _ = self.frozen_model.encoder(
                    inputs, compute_forward=True, compute_backward=False
                )
                token_embeddings = (
                    self.frozen_model.encoder.token_embedding.weight.detach()
                )
            else:
                _, hidden_states = self.frozen_model.model(
                    inputs, return_hidden_states=True
                )
                token_embeddings = (
                    self.frozen_model.model.token_embedding.weight.detach()
                )
        logits, predict_h_t_next = self.probe(
            hidden_states, batch[:, 1:], token_embeddings
        )

        MSE_loss = F.mse_loss(predict_h_t_next, hidden_states[:, 1:], reduction="mean")
        logits_loss = torch.stack(
            [
                F.cross_entropy(
                    logits[:, : -i - 1, :, i].reshape(-1, self.config.vocab_size),
                    batch[:, i + 1 :].reshape(-1),
                    reduction="mean",
                )
                for i in range(self.config.probe_depth)
            ]
        )
        total_logits_loss = logits_loss.sum()
        total_loss = MSE_loss + total_logits_loss
        # Backward pass
        if backpropagate:
            self.fabric.backward(total_loss)

        return total_loss.detach(), logits_loss.detach(), MSE_loss.detach()

    def _optimizer_modules(self) -> Tuple[nn.Module, ...]:
        return (self.probe,)

    def compile(self):
        """
        Compiles the model using torch.compile()
        """
        # Must compile before setup_fabric()
        self._assert_fabric_is_setup(setup=False)
        self.probe.compile()

    def train(self):
        """
        Set the model to training mode
        """
        self.probe.train()

    def eval(self):
        """
        Set the model to evaluation mode
        """
        self.probe.eval()
