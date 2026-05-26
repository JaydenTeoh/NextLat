import numpy as np
import pandas as pd
import torch
import lightning as L
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from datetime import datetime


class Tokenizer:
    def __init__(self):
        self.eos_token_id = 60  # there are 60 possible even permutations for 5 tokens

    def encode(self, data):
        return data  # data already in ints

    def tokenize(self, prefix):
        seq = np.array(prefix)
        return seq, len(seq)


class A5Dataset(torch.utils.data.Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
        self.permutation_len = data.shape[1]
        self.total_len = data.shape[1] + labels.shape[1]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]
        input_ids = torch.cat((x, y))
        return input_ids

    # Modularized evaluation function for A5
    def evaluate_a5(
        self,
        model,
        dataloader,
        config,
        fabric,
        show_progress_bar=True,
        generalization=False,
    ):
        """
        Evaluate A5 sequence accuracy. Returns a dict with overall and per-token accuracy.
        """
        model.eval()
        test_logs = {}
        batch_count = 0
        total_correct = torch.zeros(1, device=fabric.device)
        total_tokens = torch.zeros(1, device=fabric.device)
        tokens_correct = {}
        total_dynamics_correct = torch.zeros(1, device=fabric.device)
        total_dynamics_tokens = torch.zeros(1, device=fabric.device)
        dynamics_tokens_correct = {}
        prefix_label = "generalization/" if generalization else "val/"

        disable_pbar = True if not show_progress_bar else None
        n_test_batches = None
        if hasattr(config.trainer, "test_batches") and config.trainer.test_batches > 0:
            n_test_batches = config.trainer.test_batches // fabric.world_size

        for batch in tqdm(
            dataloader,
            desc="A5 Evaluation",
            total=n_test_batches or len(dataloader),
            leave=False,
            disable=disable_pbar,
        ):
            # Assume batch shape: (batch_size, seq_len)
            batch_size, seq_len = batch.shape
            num_target_tokens = self.permutation_len
            prefix = batch[:, :-num_target_tokens].clone()
            target = batch[:, -num_target_tokens:].clone()

            with torch.inference_mode():
                if (
                    getattr(model.model, "dynamics_model", None) is not None
                ):  # for NextLat
                    token_embeds, text_embd = model.model(
                        prefix, return_hidden_states=True
                    )
                    # Roll out P hidden states using the dynamics model:
                    # h_0 from the encoder, then h_{t+1} = dynamics(h_t, x_{t+1}) for t = 0..P-2.
                    # For A5 the convention is lm_head(h_t) predicts target[t], so we keep h_0
                    # in the prediction sequence to align with target[0].
                    current_hidden = text_embd[:, :1]
                    dynamics_hiddens = [current_hidden]
                    for i in range(num_target_tokens - 1):
                        current_hidden = model.model.dynamics_model(
                            current_hidden, token_embeds[:, i + 1 : i + 2]
                        )
                        dynamics_hiddens.append(current_hidden)
                    dynamics_hiddens = torch.cat(dynamics_hiddens, dim=1)
                    dynamics_predictions = model.model.lm_head(dynamics_hiddens).argmax(
                        dim=-1
                    )
                else:
                    dynamics_predictions = None
                predictions = model.model(prefix).argmax(dim=-1)

            correct = target.eq(predictions).float()
            completely_correct = correct.sum(dim=1).eq(num_target_tokens).float()
            total_correct += completely_correct.sum()
            total_tokens += batch_size

            # Cumulative correctness: 1s until the first miss, then 0s thereafter.
            cumulative_correct = torch.cumprod(correct, dim=1)
            per_token_acc = cumulative_correct.mean(dim=0)
            for i in range(num_target_tokens):
                if i not in tokens_correct:
                    tokens_correct[i] = torch.zeros(1, device=fabric.device)
                tokens_correct[i] += per_token_acc[i] * batch_size

            if dynamics_predictions is not None:
                dynamics_correct = target.eq(dynamics_predictions).float()
                # Cumulative correctness: 1s until the first miss, then 0s thereafter.
                # The last column equals "completely correct" for the whole sequence.
                cumulative_dynamics_correct = torch.cumprod(dynamics_correct, dim=1)
                total_dynamics_correct += cumulative_dynamics_correct[:, -1].sum()
                total_dynamics_tokens += batch_size
                per_token_dynamics_acc = cumulative_dynamics_correct.mean(dim=0)
                for i in range(num_target_tokens):
                    if i not in dynamics_tokens_correct:
                        dynamics_tokens_correct[i] = torch.zeros(
                            1, device=fabric.device
                        )
                    dynamics_tokens_correct[i] += per_token_dynamics_acc[i] * batch_size

            batch_count += 1
            if n_test_batches is not None and batch_count >= n_test_batches:
                break

        global_total_correct = fabric.all_reduce(total_correct, reduce_op="sum")
        global_total_tokens = fabric.all_reduce(total_tokens, reduce_op="sum")
        overall_accuracy = (global_total_correct / global_total_tokens).item()
        global_total_dynamics_correct = fabric.all_reduce(
            total_dynamics_correct, reduce_op="sum"
        )
        global_total_dynamics_tokens = fabric.all_reduce(
            total_dynamics_tokens, reduce_op="sum"
        )
        overall_dynamics_accuracy = (
            global_total_dynamics_correct / global_total_dynamics_tokens
        ).item()
        test_logs[f"{prefix_label}dynamics_test_accuracy"] = overall_dynamics_accuracy

        per_token_accuracy = {}
        for i in dynamics_tokens_correct:
            global_dynamics_token_correct = fabric.all_reduce(
                dynamics_tokens_correct[i], reduce_op="sum"
            )
            per_token_accuracy[f"{prefix_label}dynamics_cumulative_token_{i+1}"] = (
                global_dynamics_token_correct / global_total_dynamics_tokens
            ).item()

        for i in tokens_correct:
            global_token_correct = fabric.all_reduce(tokens_correct[i], reduce_op="sum")
            per_token_accuracy[f"{prefix_label}cumulative_token_{i+1}"] = (
                global_token_correct / global_total_tokens
            ).item()

        test_logs[f"{prefix_label}test_accuracy"] = overall_accuracy
        test_logs.update(per_token_accuracy)

        fabric.print(
            f"{datetime.now()} A5 {prefix_label}Dynamics Test Accuracy: {overall_dynamics_accuracy:.2%}"
        )

        fabric.print(
            f"{datetime.now()} A5 {prefix_label}Test Accuracy: {overall_accuracy:.2%}"
        )

        return test_logs


class A5DataModule:
    """
    PyTorch Lightning style DataModule for A5 dataset
    Merrill et al. (2025): https://arxiv.org/pdf/2404.08819
    """

    def __init__(self, fabric: L.Fabric, config):
        self.fabric = fabric
        self.batch_size = config.data.device_batch_size

        # Load data
        with open(config.data.A5_data_path, "r") as f:
            df = pd.read_csv(f)
            input_array = df["input"].str.split(" ", expand=True).astype(int).to_numpy()
            target_array = (
                df["target"].str.split(" ", expand=True).astype(int).to_numpy()
            )

        data = torch.tensor(input_array, dtype=torch.long)
        labels = torch.tensor(target_array, dtype=torch.long)

        # Split the data into training and validation sets
        train_data, val_data, train_labels, val_labels = train_test_split(
            data, labels, test_size=0.2, random_state=42
        )

        # No-op tokenizer
        self.tokenizer = Tokenizer()

        self.train_dataset = A5Dataset(train_data, train_labels)
        self.val_dataset = A5Dataset(val_data, val_labels)

        if config.data.test_generalization:
            # Load data
            with open(config.data.A5_generalization_data_path, "r") as f:
                df = pd.read_csv(f)
                input_array = (
                    df["input"].str.split(" ", expand=True).astype(int).to_numpy()
                )
                target_array = (
                    df["target"].str.split(" ", expand=True).astype(int).to_numpy()
                )

            generalization_data = torch.tensor(input_array, dtype=torch.long)
            generalization_labels = torch.tensor(target_array, dtype=torch.long)

            # Split the data into training and validation sets
            _, gen_val_data, _, gen_val_labels = train_test_split(
                generalization_data,
                generalization_labels,
                test_size=0.2,
                random_state=42,
            )

            self.generalization_dataset = A5Dataset(gen_val_data, gen_val_labels)
            self.gen_total_len = (
                generalization_data.shape[1] + generalization_labels.shape[1]
            )

        self.vocab_size = 60  # Total vocabulary size
        self.permutation_len = data.shape[1]
        print(f"Permutation length: {self.permutation_len}")
        self.total_len = data.shape[1] + labels.shape[1]
        print(f"Total length: {self.total_len}")

    def update_config(self, config):
        config.model.vocab_size = self.vocab_size
        # we don't mask targets since the targets is different from input
        config.model.context_length = 0

    def get_tokenizer(self):
        return self.tokenizer

    def prepare_batch(self, batch):
        return batch

    def train_dataloader(self):
        dataloader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            collate_fn=self.collate_fn,
        )
        # This will automatically partition data between devices
        return self.fabric.setup_dataloaders(dataloader, use_distributed_sampler=True)

    def val_dataloader(self):
        dataloader = torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=True,
            collate_fn=self.collate_fn,
        )
        # This will automatically partition data between devices
        return self.fabric.setup_dataloaders(dataloader, use_distributed_sampler=True)

    def generalization_dataloader(self):
        dataloader = torch.utils.data.DataLoader(
            self.generalization_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=True,
            collate_fn=self.collate_fn,
        )
        return self.fabric.setup_dataloaders(dataloader, use_distributed_sampler=True)

    @staticmethod
    def collate_fn(batch):
        x_batch = torch.stack([item for item in batch])
        return x_batch
