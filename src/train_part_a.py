import argparse
import torch
import pandas as pd

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq

from src.utils.io import load_yaml, ensure_dir, save_csv
from src.utils.seed import seed_everything
from src.data.opus100 import load_opus100_en_nl
from src.models.part_a_lightning import Seq2SeqLightningModule


def tokenize_seq2seq(ds, tokenizer, max_src_len: int, max_tgt_len: int):
    def preprocess(batch):
        inputs = tokenizer(
            batch["src"],
            max_length=max_src_len,
            truncation=True,
            padding="max_length",
        )
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                batch["tgt"],
                max_length=max_tgt_len,
                truncation=True,
                padding="max_length",
            )
        inputs["labels"] = labels["input_ids"]
        return inputs

    return ds.map(preprocess, batched=True, remove_columns=["src", "tgt"])


def main(config_path: str):
    cfg = load_yaml(config_path)
    seed_everything(cfg.get("seed", 42))

    model_name = cfg["model_name"]
    out_dir = cfg["output_dir"]
    ensure_dir(out_dir)