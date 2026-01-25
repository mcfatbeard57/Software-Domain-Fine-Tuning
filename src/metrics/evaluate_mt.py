import evaluate

_bleu = evaluate.load("sacrebleu")
_chrf = evaluate.load("chrf")

def compute_bleu_chrf(preds, refs):
    bleu_score = _bleu.compute(predictions=preds, references=[[r] for r in refs])["score"]
    chrf_score = _chrf.compute(predictions=preds, references=refs)["score"]
    return {"BLEU": float(bleu_score), "chrF": float(chrf_score)}
