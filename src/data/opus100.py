from datasets import load_dataset

def load_opus100_en_nl(train_rows: int, val_rows: int, seed: int = 42):
    ds = load_dataset("opus100", "en-nl")["train"].shuffle(seed=seed)

    val = ds.select(range(min(val_rows, len(ds))))
    train = ds.select(range(min(val_rows, len(ds)), min(val_rows + train_rows, len(ds))))

    def to_src_tgt(ex):
        return {"src": ex["translation"]["en"], "tgt": ex["translation"]["nl"]}

    train = train.map(to_src_tgt, remove_columns=train.column_names)
    val = val.map(to_src_tgt, remove_columns=val.column_names)

    return train, val
