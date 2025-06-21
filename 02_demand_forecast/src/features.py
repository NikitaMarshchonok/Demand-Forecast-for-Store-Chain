import pandas as pd
import numpy as np
from pathlib import Path

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
OUT = Path(__file__).resolve().parents[1] / "data" / "features"
OUT.mkdir(parents=True, exist_ok=True)

def build_features():
    sales    = pd.read_csv(RAW / "sales_train_validation.csv")
    calendar = pd.read_csv(RAW / "calendar.csv")
    day_cols = [c for c in sales.columns if c.startswith("d_")]

    df = (sales
          .melt(id_vars=["id"],
                value_vars=day_cols,
                var_name="d",
                value_name="sales")
          .merge(calendar[["d","date","wday","month","year"]],
                 on="d", how="left")
          .sort_values(["id","date"]))

    df["lag_1"]  = df.groupby("id")["sales"].shift(1)
    df["lag_7"]  = df.groupby("id")["sales"].shift(7)
    df["rmean_7"]  = (df.groupby("id")["lag_1"]
                        .rolling(7).mean()
                        .reset_index(level=0, drop=True))
    df["rmean_28"] = (df.groupby("id")["lag_1"]
                        .rolling(28).mean()
                        .reset_index(level=0, drop=True))
    df["date"] = pd.to_datetime(df["date"])
    df.dropna(inplace=True)

    df.to_parquet(OUT / "features.parquet")
    print("Features saved:", OUT / "features.parquet")

if __name__ == "__main__":
    build_features()
