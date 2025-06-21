import lightgbm as lgb, joblib, json
import pandas as pd, numpy as np
from datetime import timedelta
from pathlib import Path

BASE   = Path(__file__).resolve().parents[1]
DATA   = BASE / "data"
MODEL  = BASE / "models"
MODEL.mkdir(exist_ok=True)

def main():
    df = pd.read_parquet(DATA / "features" / "features.parquet")
    last_date = df["date"].max()
    val_start = last_date - timedelta(days=27)

    train_df = df[df["date"] <  val_start]
    valid_df = df[df["date"] >= val_start]

    feats = ["lag_1","lag_7","rmean_7","rmean_28","wday","month","year"]
    X_tr, y_tr = train_df[feats], train_df["sales"]
    X_va, y_va = valid_df[feats], valid_df["sales"]

    dtrain, dval = lgb.Dataset(X_tr, y_tr), lgb.Dataset(X_va, y_va)
    params = dict(objective="regression", metric="rmse",
                  learning_rate=0.1, num_leaves=31,
                  feature_fraction=0.8, verbose=-1)

    model = lgb.train(params, dtrain,
                      num_boost_round=500,
                      valid_sets=[dval],
                      callbacks=[lgb.early_stopping(50)])

    y_pred = model.predict(X_va, num_iteration=model.best_iteration)
    rmse   = float(np.sqrt(((y_va - y_pred) ** 2).mean()))

    model.save_model(MODEL / "lgbm_m5.txt")
    joblib.dump(model, MODEL / "lgbm_m5.pkl")
    json.dump({"rmse": rmse}, open(MODEL / "metrics.json", "w"))
    print("RMSE:", rmse)

if __name__ == "__main__":
    main()
