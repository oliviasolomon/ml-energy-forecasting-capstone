import os
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "electricity_long_2017.parquet"
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"

MODELS_DIR.mkdir(exist_ok=True)
OUTPUTS_DIR.mkdir(exist_ok=True)


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    ts = df["timestamp"]
    df["year"] = ts.dt.year
    df["month"] = ts.dt.month
    df["day"] = ts.dt.day
    df["hour"] = ts.dt.hour
    df["dayofweek"] = ts.dt.dayofweek
    df["dayofyear"] = ts.dt.dayofyear
    df["weekofyear"] = ts.dt.isocalendar().week.astype(int)

    # cyclic encodings for hour, day-of-week, and day-of-year
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)
    df["doy_sin"] = np.sin(2 * np.pi * df["dayofyear"] / 365)
    df["doy_cos"] = np.cos(2 * np.pi * df["dayofyear"] / 365)

    return df

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # floor area intensity
    df["sqft"] = df["sqft"].astype(float)
    df["meter_per_sqft"] = df["meter_reading"] / df["sqft"].replace(0, np.nan)

    # time features
    df = add_time_features(df)

    # weekend flag
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)

    # weather rolling 24h and 72h features
    df = df.sort_values("timestamp")
    for col in ["airTemperature", "dewTemperature", "windSpeed"]:
        if col in df.columns:
            df[f"{col}_roll24_mean"] = (
                df.groupby("building_id")[col]
                .transform(lambda s: s.rolling(24, min_periods=1).mean())
            )
            df[f"{col}_roll72_mean"] = (
                df.groupby("building_id")[col]
                .transform(lambda s: s.rolling(72, min_periods=1).mean())
            )

    return df

def get_train_test():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"{DATA_PATH} not found. Run src.data_preprocessing first."
        )

    df = pd.read_parquet(DATA_PATH)
    df = build_features(df)

    # split by year
    train_df = df[df["timestamp"].dt.year == 2016].copy()
    test_df = df[df["timestamp"].dt.year == 2017].copy()

    return train_df, test_df

def train_lgbm(train_df: pd.DataFrame, target_col: str, model_path: Path):
    # log transform targets
    y = train_df[target_col].clip(lower=0)
    y_log = np.log1p(y)
  
    # drop raw targets and timestamp
    drop_cols = [
        "meter_reading",
        "meter_per_sqft",
        "timestamp",
        "primary_space_usage",
    ]
    X = train_df.drop(columns=[c for c in drop_cols if c in train_df.columns])

    # building_id and primary_space_usage if present
    cat_cols = []
    if "building_id" in X.columns:
        cat_cols.append("building_id")
    if "primary_space_usage" in train_df.columns and \
            "primary_space_usage" in X.columns:
        cat_cols.append("primary_space_usage")

    # train/valid split last 30 days of 2016 as validation
    cutoff = train_df["timestamp"].max() - pd.Timedelta(days=30)
    train_mask = train_df["timestamp"] < cutoff
    valid_mask = ~train_mask

    X_train, y_train = X[train_mask], y_log[train_mask]
    X_valid, y_valid = X[valid_mask], y_log[valid_mask]

    train_ds = lgb.Dataset(
        X_train,
        label=y_train,
        categorical_feature=cat_cols,
        free_raw_data=False,
    )
    valid_ds = lgb.Dataset(
        X_valid,
        label=y_valid,
        categorical_feature=cat_cols,
        free_raw_data=False,
    )

    params = {
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": 0.05,
        "num_leaves": 64,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "min_data_in_leaf": 50,
        "verbosity": -1,
    }

    model = lgb.train(
        params,
        train_ds,
        num_boost_round=2000,
        valid_sets=[train_ds, valid_ds],
        valid_names=["train", "valid"],
        callbacks=[lgb.early_stopping(stopping_rounds=100)],
    )

    model.save_model(str(model_path))
    return model, X.columns.tolist()


def main():
    train_df, test_df = get_train_test()

    # model 1: log(meter)
    model1_path = MODELS_DIR / "lgbm_y_log.txt"
    model1, feature_cols = train_lgbm(train_df, "meter_reading", model1_path)

    # model 2: log(meter_per_sqft)
    model2_path = MODELS_DIR / "lgbm_y_log_per_sqft.txt"
    model2, _ = train_lgbm(train_df, "meter_per_sqft", model2_path)

    # prepare test features
    test_features = test_df[feature_cols].copy()

    # predict 2017 data
    pred1 = np.expm1(model1.predict(test_features))
    pred2 = np.expm1(model2.predict(test_features)) * test_df["sqft"].values

    preds = 0.5 * (pred1 + pred2)

    out = test_df[["timestamp", "building_id"]].copy()
    out["actual"] = test_df["meter_reading"].values
    out["predicted"] = preds

    OUTPUTS_DIR.mkdir(exist_ok=True)
    out.to_csv(
        OUTPUTS_DIR / "electricity_2017_predictions_long.csv",
        index=False,
    )
    print("Saved predictions to outputs/electricity_2017_predictions_long.csv")


if __name__ == "__main__":
    main()
