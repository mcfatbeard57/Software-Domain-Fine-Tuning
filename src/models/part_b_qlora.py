import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

def load_qlora_model(model_name: str, qlora_cfg: dict, lora_cfg: dict):
    bnb = BitsAndBytesConfig(
        load_in_4bit=qlora_cfg.get("load_in_4bit", True),
        bnb_4bit_use_double_quant=qlora_cfg.get("bnb_4bit_use_double_quant", True),
        bnb_4bit_quant_type=qlora_cfg.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_compute_dtype=getattr(torch, qlora_cfg.get("bnb_4bit_compute_dtype", "float16")),
    )

    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    
    # ✅ ADD THIS LINE (decoder-only generation wants left padding)
    tok.padding_side = "left"

    # ✅ keep this (pad token required for batching)

    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb,
        device_map="auto"
    )

    model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False  # required if using gradient checkpointing

    lora = LoraConfig(
        r=lora_cfg.get("r", 16),
        lora_alpha=lora_cfg.get("alpha", 32),
        lora_dropout=lora_cfg.get("dropout", 0.05),
        bias=lora_cfg.get("bias", "none"),
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora)
    return model, tok
