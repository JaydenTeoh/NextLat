# Data Generation

> All commands below should be run from the **root of the repository**.

---

## Manhattan Taxi Rides

We use the *random walks* dataset from ["Evaluating the World Model Implicit in a Generative Model"](https://arxiv.org/abs/2406.03689). For convenience, we host a pretokenized version on Hugging Face (tokenized with the custom `SimpleTokenizer` in `data/manhattan_dataset.py`). By default, the scripts in this repository will train using the pretokenized dataset.

For reproducibility, we also provide a script to download the raw training data directly from the authors' [Google Drive](https://drive.google.com/drive/folders/1gP4EiLqBASu89vSk62JvOwfVYSimaHSY) using `gdown`:

```bash
./data/manhattan/random_walks/download_random_walks_gdrive.sh
```
**Note:** To use the raw datasets, set `config.data.use_pretokenized: false` in the training config.  
We do not recommend this path for routine runs because tokenizing the full dataset from scratch in `data/manhattan_dataset.py` is really slow.

---

## Countdown

We modified the data generation setup from ["Stream of Search (SoS): Learning to Search in Language"](https://arxiv.org/abs/2404.03683).

- **Countdown with 4 input numbers:**

```bash
python data/countdown/generate.py --seed 444 --min_range 4 --start_range 4 --num_samples 500000
```

- **Countdown with 5 input numbers:**

```bash
python data/countdown/generate.py --seed 444 --min_range 5 --start_range 5 --num_samples 500000
```

**Note:** Generating the data will take a while. We only evaluate on Countdown with 4 input numbers in our paper.

---

## Path-Star Graph

We follow the setup from ["The Pitfalls of Next-Token Prediction"](https://arxiv.org/abs/2403.06963). To generate the data for each graph configuration, run:

- **G(2, 5):**

```bash
python data/stargraph/prepare.py --num_samples 200000 --num_paths 2 --path_length 10 --max_nodes 100 --generate_test_data
```

- **G(5, 5):**

```bash
python data/stargraph/prepare.py --num_samples 200000 --num_paths 5 --path_length 5 --max_nodes 100 --generate_test_data
```

- **G(7, 7):**

```bash
python data/stargraph/prepare.py --num_samples 200000 --num_paths 7 --path_length 7 --max_nodes 100 --generate_test_data
```


Each command creates a data file (e.g., `graph_2_5_sample_1000000.txt`) in `data/stargraph/`.

---

## TinyStories

We follow the setup from ["The Belief State Transformer"](https://arxiv.org/abs/2410.23506).

No data preparation is necessary. The `TinyStoriesDataModule` class (see `data/tinystories.py`) fetches the data directly from HuggingFace.

---

## State Tracking

For the A5 state tracking dataset, run:

```bash
python data/a5/prepare.py --group A5 --k 12 --samples 1000000 --seed 444
python data/a5/prepare.py --group A5 --k 36 --samples 1000000 --seed 444
```
- `--k` determines the length of the sequences. 

---

## FineWeb-Edu

We pretokenize the [FineWeb-Edu](https://arxiv.org/abs/2406.17557) dataset (~100B tokens), streamed from [Hugging Face](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu), using the GPT-2 tokenizer:

```bash
python data/fineweb/pretokenize_finewebedu.py \
  --subset sample-100BT \
  --output_dir data/fineweb/pretokenized-100BT \
  --shard_tokens 100000000 \
  --num_workers 96 \
  --text_batch_size 1024 \
  --expected_total_chunks 1000
```

Tune `--num_workers` to your available CPU count.

**Note:** This will take several hours on a multi-core CPU, so we recommend storing the tokenized output on a persistent data mount. To do a smoke test before the full run, set `--max_documents` or `--max_tokens` (e.g., see `/scripts/fineweb/pretokenize_finewebedu_10BT.sh` for pretokenizing a 10B token subset).
