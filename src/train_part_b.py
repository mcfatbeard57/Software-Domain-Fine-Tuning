import argparse
import yaml
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
from src.utils.seed import seed_everything  
from src.utils.io import load_yaml, ensure_dir
from src.data.opus100 import load_opus100_en_nl
from src.models.part_b_qlora import load_qlora_model

def build_prompt(src: str) -> str:
    return (
        "You are a professional software localization translator.\n"
        "Translate the following English text into fluent Dutch.\n"
        "Return ONLY the Dutch translation.\n\n"
        f"English: {src}\nDutch:"
    )
def preprocess_decoder_only(batch, tokenizer, max_len: int):
    prompts, full_texts = [], []
    for s, t in zip(batch["src"], batch["tgt"]):
        p = build_prompt(s)
        prompts.append(p)
        full_texts.append(p + " " + t)
    tok_full = tokenizer(full_texts, max_length=max_len, truncation=True, padding="max_length")
    tok_prompt = tokenizer(prompts, max_length=max_len, truncation=True, padding="max_length")
    labels = []
    for i in range(len(tok_full["input_ids"])):
        prompt_len = sum(tok_prompt["attention_mask"][i])
        lab = tok_full["input_ids"][i].copy()
        for j in range(prompt_len):
            lab[j] = -100
        labels.append(lab)
    tok_full["labels"] = labels
    return tok_full

def main(config_path: str):
    cfg = load_yaml(config_path)
    seed_everything(cfg.get("seed", 42))

    out_dir = cfg["output_dir"]
    ensure_dir(out_dir)

    train_ds, val_ds = load_opus100_en_nl(cfg["train_rows"], cfg["val_rows"], seed=cfg.get("seed", 42))
    model, tok = load_qlora_model(cfg["model_name"], cfg["qlora"], cfg["lora"])

    if cfg["train"].get("gradient_checkpointing", False):
        use_reentrant = cfg["train"].get("use_reentrant", False)
        try:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": use_reentrant})
        except TypeError:
            model.gradient_checkpointing_enable()

    train_tok = train_ds.map(lambda b: preprocess_decoder_only(b, tok, cfg["max_len"]),
                             batched=True, remove_columns=["src", "tgt"])
    val_tok = val_ds.map(lambda b: preprocess_decoder_only(b, tok, cfg["max_len"]),
                         batched=True, remove_columns=["src", "tgt"])
    
    
    
    collator = DataCollatorForLanguageModeling(tok, mlm=False)

    args = TrainingArguments(
        output_dir=out_dir,
        num_train_epochs=cfg["train"]["epochs"],
        learning_rate=cfg["train"]["lr"],
        per_device_train_batch_size=cfg["train"]["per_device_train_bs"],
        per_device_eval_batch_size=cfg["train"]["per_device_eval_bs"],
        gradient_accumulation_steps=cfg["train"]["grad_accum"],
        evaluation_strategy="steps",
        eval_steps=cfg["train"]["eval_steps"],
        save_steps=cfg["train"]["save_steps"],
        save_total_limit=cfg["train"]["save_total_limit"],
        fp16=cfg["train"]["fp16"],
        logging_steps=50,
        report_to="none",
    )

  
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_tok,
        eval_dataset=val_tok,
        data_collator=collator,
        tokenizer=tok,
    )
    
    trainer.train()
    trainer.save_model(out_dir)
    tok.save_pretrained(out_dir)
    print("✅ Saved Part B QLoRA model to:", out_dir)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    main(args.config)