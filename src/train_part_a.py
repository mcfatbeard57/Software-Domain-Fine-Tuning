# src/train_part_a.py

import argparse
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
)

from src.utils.io import load_yaml, ensure_dir
from src.utils.seed import seed_everything
from src.data.opus100 import load_opus100_en_nl
from src.models.part_a_lightning import Seq2SeqLightningModule


def _to_int(x, name: str) -> int:
    try:
        return int(x)
    except Exception as e:
        raise ValueError(f"Config field '{name}' must be int-like. Got {x} ({type(x)}).") from e


def _to_float(x, name: str) -> float:
    try:
        return float(x)
    except Exception as e:
        raise ValueError(f"Config field '{name}' must be float-like. Got {x} ({type(x)}).") from e


def tokenize_seq2seq(ds, tokenizer, max_src_len: int, max_tgt_len: int):
    """
    Uses the modern HF API: tokenizer(..., text_target=...)
    This avoids deprecated as_target_tokenizer warnings.
    """
    def preprocess(batch):
        return tokenizer(
            batch["src"],
            text_target=batch["tgt"],
            max_length=max_src_len,
            truncation=True,
            padding="max_length",
        )

    return ds.map(preprocess, batched=True, remove_columns=["src", "tgt"])


def main(config_path: str):
    cfg = load_yaml(config_path)
    seed_everything(_to_int(cfg.get("seed", 42), "seed"))

    model_name = cfg["model_name"]
    out_dir = cfg["output_dir"]
    ensure_dir(out_dir)

    # ---- Parse + cast training config safely ----
    train_cfg = cfg["train"]

    epochs = _to_int(train_cfg["epochs"], "train.epochs")
    train_bs = _to_int(train_cfg["train_batch_size"], "train.train_batch_size")
    val_bs = _to_int(train_cfg["val_batch_size"], "train.val_batch_size")

    lr = _to_float(train_cfg["lr"], "train.lr")
    weight_decay = _to_float(train_cfg["weight_decay"], "train.weight_decay")
    warmup_ratio = _to_float(train_cfg["warmup_ratio"], "train.warmup_ratio")

    gradient_clip_val = _to_float(train_cfg["gradient_clip_val"], "train.gradient_clip_val")
    log_every_n_steps = _to_int(train_cfg["log_every_n_steps"], "train.log_every_n_steps")

    # Precision is a string like "16-mixed" in config; safe as-is
    precision_cfg = train_cfg.get("precision", "16-mixed")

    # ---- Load model/tokenizer ----
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    # ---- Load data ----
    train_rows = _to_int(cfg["train_rows"], "train_rows")
    val_rows = _to_int(cfg["val_rows"], "val_rows")

    train_ds, val_ds = load_opus100_en_nl(train_rows, val_rows, seed=_to_int(cfg.get("seed", 42), "seed"))

    max_src_len = _to_int(cfg["max_src_len"], "max_src_len")
    max_tgt_len = _to_int(cfg["max_tgt_len"], "max_tgt_len")

    train_tok = tokenize_seq2seq(train_ds, tokenizer, max_src_len, max_tgt_len)
    val_tok = tokenize_seq2seq(val_ds, tokenizer, max_src_len, max_tgt_len)

    collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    def make_loader(ds, batch_size, shuffle=False):
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=collator,
            num_workers=2,
            pin_memory=torch.cuda.is_available(),
        )

    train_loader = make_loader(train_tok, train_bs, shuffle=True)
    val_loader = make_loader(val_tok, val_bs, shuffle=False)

    # ---- Lightning module ----
    lm = Seq2SeqLightningModule(
        model=model,
        lr=lr,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
    )

    # ---- Trainer ----
    use_gpu = torch.cuda.is_available()
    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator="gpu" if use_gpu else "cpu",
        devices=1,
        precision=precision_cfg if use_gpu else 32,
        gradient_clip_val=gradient_clip_val,
        log_every_n_steps=log_every_n_steps,
    )

    trainer.fit(lm, train_loader, val_loader)

    # ---- Save outputs ----
    lm.model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    print(f"✅ Saved Part A model to: {out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    main(args.config)