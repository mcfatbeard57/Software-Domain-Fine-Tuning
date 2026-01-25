# src/data/flores.py

from typing import Tuple, List
import os
import urllib.request
from urllib.error import HTTPError

from datasets import load_dataset, Dataset

FLORES_REPO = "facebook/flores"

# IMPORTANT:
# HF "main" does NOT contain the parquet language folders (eng_Latn/, nld_Latn/, etc.) right now.
# Parquet files exist in commit snapshots created during parquet conversion.
# We'll try a list of revisions until we find one that has the files.
REVISION_CANDIDATES = [
    # Known snapshot where eng_Latn parquet exists (from HF file browser snapshots)
    "982199a7751a9e75fd46d17887827432661fdf20",
    # Another parquet snapshot often referenced in file URLs during conversion
    "65213e6e50bc8eb4b84f3f2b0c3be34445f3de5c",
    # Keep "main" last (will 404 for parquet files, but harmless to try)
    "main",
]

CACHE_DIR = os.environ.get("FLORES_CACHE_DIR", "/content/flores_cache")


def _download(url: str, local_path: str) -> None:
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    urllib.request.urlretrieve(url, local_path)


def _try_download_parquet(lang: str, split: str) -> Tuple[str, str]:
    """
    Try downloading the parquet file for (lang, split) across candidate revisions.
    Returns: (local_path, revision_used)
    """
    fname = f"{split}-00000-of-00001.parquet"

    last_err = None
    for rev in REVISION_CANDIDATES:
        url = f"https://huggingface.co/datasets/{FLORES_REPO}/resolve/{rev}/{lang}/{fname}"
        local_path = os.path.join(CACHE_DIR, f"{rev}_{lang}_{fname}")

        # If already downloaded, reuse
        if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
            return local_path, rev

        try:
            print(f"Downloading FLORES parquet (rev={rev}):\n  {url}\n-> {local_path}")
            _download(url, local_path)
            # basic sanity check
            if os.path.getsize(local_path) > 0:
                return local_path, rev
        except HTTPError as e:
            last_err = f"{rev}: HTTP {e.code}"
            # remove empty partial file if created
            try:
                if os.path.exists(local_path) and os.path.getsize(local_path) == 0:
                    os.remove(local_path)
            except Exception:
                pass
            continue
        except Exception as e:
            last_err = f"{rev}: {type(e).__name__}: {e}"
            try:
                if os.path.exists(local_path) and os.path.getsize(local_path) == 0:
                    os.remove(local_path)
            except Exception:
                pass
            continue

    raise RuntimeError(
        f"Could not download FLORES parquet for {lang}/{split}. "
        f"Tried revisions: {REVISION_CANDIDATES}. Last error: {last_err}"
    )


def _load_flores_parquet_local(lang: str, split: str) -> Dataset:
    """
    Bulletproof Colab loader:
    - downloads parquet locally from HF resolve/<revision>/...
    - loads local parquet via datasets parquet loader
    """
    local_path, rev = _try_download_parquet(lang, split)
    ds = load_dataset("parquet", data_files=local_path, split="train")
    # attach metadata for debugging
    ds = ds.add_column("_flores_revision", [rev] * len(ds))
    return ds


def _align_en_nl(en_ds: Dataset, nl_ds: Dataset) -> Dataset:
    """
    Align EN and NL row-wise for a given split.
    FLORES parquet per-lang contains a `sentence` column.
    """
    if "sentence" not in en_ds.column_names:
        raise ValueError(f"Expected 'sentence' in EN dataset. Found: {en_ds.column_names}")
    if "sentence" not in nl_ds.column_names:
        raise ValueError(f"Expected 'sentence' in NL dataset. Found: {nl_ds.column_names}")

    en_sent = en_ds["sentence"]
    nl_sent = nl_ds["sentence"]
    n = min(len(en_sent), len(nl_sent))

    # keep revision metadata (useful in logs)
    en_rev = en_ds["_flores_revision"][0] if "_flores_revision" in en_ds.column_names and len(en_ds) else "unknown"
    nl_rev = nl_ds["_flores_revision"][0] if "_flores_revision" in nl_ds.column_names and len(nl_ds) else "unknown"

    return Dataset.from_dict(
        {
            "src": en_sent[:n],
            "tgt": nl_sent[:n],
            "_en_revision": [en_rev] * n,
            "_nl_revision": [nl_rev] * n,
        }
    )


def load_flores_en_nl() -> Tuple[Dataset, Dataset]:
    """
    Returns:
      flores_dev, flores_devtest
    with columns: src (English), tgt (Dutch)

    Script-free + Colab-friendly.
    """
    errors: List[tuple] = []

    try:
        en_dev = _load_flores_parquet_local("eng_Latn", "dev")
        nl_dev = _load_flores_parquet_local("nld_Latn", "dev")
        en_devtest = _load_flores_parquet_local("eng_Latn", "devtest")
        nl_devtest = _load_flores_parquet_local("nld_Latn", "devtest")

        flores_dev = _align_en_nl(en_dev, nl_dev)
        flores_devtest = _align_en_nl(en_devtest, nl_devtest)
        return flores_dev, flores_devtest
    except Exception as e:
        errors.append(("flores(parquet_local)", str(e)))

    raise RuntimeError(f"Could not load FLORES EN-NL. Errors: {errors}")
