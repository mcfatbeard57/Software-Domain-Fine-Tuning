import pandas as pd

def load_xlsx_testset(xlsx_path: str,
                      src_col: str = "English Source",
                      tgt_col: str = "Reference Translation"):
    df = pd.read_excel(xlsx_path)
    df = df.rename(columns={src_col: "src", tgt_col: "tgt"})
    df = df.dropna(subset=["src", "tgt"]).reset_index(drop=True)
    return df