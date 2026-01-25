import argparse
import os
import torch
import pandas as pd

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM

from src.utils.io import load_yaml, ensure_dir, save_csv
from src.metrics.evaluate_mt import compute_bleu_chrf
from src.data.flores import load_flores_en_nl
from src.data.xlsx_loader import load_xlsx_testset


@torch.no_grad()
def generate_seq2seq(model, tokenizer, sources, max_new_tokens=128, batch_size=16, max_src_len=128):
    model.eval()
    preds = []
    for i in range(0, len(sources), batch_size):
        batch = sources[i:i+batch_size]
        enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_src_len).to(model.device)
        gen = model.generate(**enc, max_new_tokens=max_new_tokens, num_beams=4)
        out = tokenizer.batch_decode(gen, skip_special_tokens=True)
        preds.extend(out)
    return preds


@torch.no_grad()
def generate_decoder_only(model, tokenizer, sources, max_new_tokens=128, batch_size=8, max_len=256):
    model.eval()
    preds = []

    def build_prompt(src: str) -> str:
        return (
            "You are a professional software localization translator.\n"
            "Translate the following English text into fluent Dutch.\n"
            "Return ONLY the Dutch translation.\n\n"
            f"English: {src}\nDutch:"
        )

    for i in range(0, len(sources), batch_size):
        batch_src = sources[i:i+batch_size]
        prompts = [build_prompt(s) for s in batch_src]
        enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=max_len).to(model.device)

        gen = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=4,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id
        )
        decoded = tokenizer.batch_decode(gen, skip_special_tokens=True)
        for text in decoded:
            if "Dutch:" in text:
                pred = text.split("Dutch:", 1)[1].strip()
            else:
                pred = text.strip()
            preds.append(pred)

    return preds


def main(config_a: str, config_b: str, xlsx_path: str, run_part_b: bool):
    cfg_a = load_yaml(config_a)
    cfg_b = load_yaml(config_b)

    out_root = "outputs"
    ensure_dir(out_root)

    # Load eval sets
    flores_dev, flores_devtest = load_flores_en_nl()
    df_xlsx = load_xlsx_testset(xlsx_path)
    xlsx_src = df_xlsx["src"].tolist()
    xlsx_ref = df_xlsx["tgt"].tolist()

    flores_src = [x["src"] for x in flores_devtest]
    flores_ref = [x["tgt"] for x in flores_devtest]

    results = []

    # -----------------------
    # Part A: Evaluate
    # -----------------------
    a_dir = cfg_a["output_dir"]
    a_tok = AutoTokenizer.from_pretrained(a_dir)
    a_model = AutoModelForSeq2SeqLM.from_pretrained(a_dir).to("cuda" if torch.cuda.is_available() else "cpu")

    flores_pred_a = generate_seq2seq(a_model, a_tok, flores_src, max_src_len=cfg_a["max_src_len"])
    xlsx_pred_a = generate_seq2seq(a_model, a_tok, xlsx_src, max_src_len=cfg_a["max_src_len"])

    m_flo_a = compute_bleu_chrf(flores_pred_a, flores_ref)
    m_xl_a = compute_bleu_chrf(xlsx_pred_a, xlsx_ref)

    results.append({"model": "encoder-decoder (Part A)", "dataset": "flores-devtest", **m_flo_a})
    results.append({"model": "encoder-decoder (Part A)", "dataset": "xlsx_software_test", **m_xl_a})

    save_csv(pd.DataFrame({"src": flores_src, "ref": flores_ref, "pred": flores_pred_a}),
             os.path.join(out_root, "preds_encoder_decoder_flores.csv"))
    save_csv(pd.DataFrame({"src": xlsx_src, "ref": xlsx_ref, "pred": xlsx_pred_a}),
             os.path.join(out_root, "preds_encoder_decoder_xlsx.csv"))

    # -----------------------
    # Part B: Evaluate (optional)
    # -----------------------
    if run_part_b:
        b_dir = cfg_b["output_dir"]
        b_tok = AutoTokenizer.from_pretrained(b_dir)
        b_model = AutoModelForCausalLM.from_pretrained(b_dir, device_map="auto")

        flores_pred_b = generate_decoder_only(b_model, b_tok, flores_src[:500], max_len=cfg_b["max_len"])
        m_flo_b = compute_bleu_chrf(flores_pred_b, flores_ref[:500])
        results.append({"model": "decoder-only (Part B)", "dataset": "flores-devtest (subset500)", **m_flo_b})

        xlsx_pred_b = generate_decoder_only(b_model, b_tok, xlsx_src, max_len=cfg_b["max_len"])
        m_xl_b = compute_bleu_chrf(xlsx_pred_b, xlsx_ref)
        results.append({"model": "decoder-only (Part B)", "dataset": "xlsx_software_test", **m_xl_b})

        save_csv(pd.DataFrame({"src": flores_src[:500], "ref": flores_ref[:500], "pred": flores_pred_b}),
                 os.path.join(out_root, "preds_decoder_only_flores_subset.csv"))
        save_csv(pd.DataFrame({"src": xlsx_src, "ref": xlsx_ref, "pred": xlsx_pred_b}),
                 os.path.join(out_root, "preds_decoder_only_xlsx.csv"))

    df_results = pd.DataFrame(results)
    save_csv(df_results, os.path.join(out_root, "metrics_summary.csv"))
    print("\n✅ Metrics:\n", df_results)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config_a", required=True)
    ap.add_argument("--config_b", required=True)
    ap.add_argument("--xlsx_path", required=True)
    ap.add_argument("--run_part_b", action="store_true", help="Evaluate Part B (requires Part B model already trained).")
    args = ap.parse_args()


    main(args.config_a, args.config_b, args.xlsx_path, args.run_part_b)
