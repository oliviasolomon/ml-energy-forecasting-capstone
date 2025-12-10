from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lightgbm as lgb

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "electricity_long_2017.parquet"
PRED_PATH = BASE_DIR / "outputs" / "electricity_2017_predictions_long.csv"
MODEL_Y_LOG_PATH = BASE_DIR / "models" / "lgbm_y_log.txt"

FIG_DIR = BASE_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)

# load merged 2017 data
def load_merged_2017() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"{DATA_PATH} not found. Run data_preprocessing.py first."
        )
    if not PRED_PATH.exists():
        raise FileNotFoundError(
            f"{PRED_PATH} not found. Run train_and_predict.py first."
        )

    # processed long-format actuals + metadata
    df_all = pd.read_parquet(DATA_PATH)
    df_all["building_id"] = df_all["building_id"].astype(str).str.strip()

    # only keep necessary columns from the parquet file
    meta_cols = ["timestamp", "building_id", "primary_space_usage", "sqft"]
    df_meta = df_all[meta_cols].drop_duplicates()

    # predictions file only 2017
    preds = pd.read_csv(PRED_PATH, parse_dates=["timestamp"])
    preds["building_id"] = preds["building_id"].astype(str).str.strip()
    preds_2017 = preds[preds["timestamp"].dt.year == 2017].copy()

    # merge predictions with metadata
    df = preds_2017.merge(
        df_meta,
        on=["timestamp", "building_id"],
        how="left",
    )

    # ensure actual column exists
    if "actual" not in df.columns:
        df_all_actual = df_all[["timestamp", "building_id", "meter_reading"]]
        df = df.merge(
            df_all_actual,
            on=["timestamp", "building_id"],
            how="left",
            suffixes=("", "_meter"),
        )
        df["actual"] = df["meter_reading"]

    return df

# predicted vs actual week of Oct 21–28, 2017 (3 stacked subplots)
def make_week_oct21_28_plot(df: pd.DataFrame) -> None:
    start = pd.Timestamp("2017-10-21")
    end = pd.Timestamp("2017-10-28")

    mask = (df["timestamp"] >= start) & (df["timestamp"] < end)
    df_week = df[mask].copy()

    def agg_by_usage(usage_label: str) -> pd.DataFrame:
        sub = df_week[df_week["primary_space_usage"] == usage_label]
        return (
            sub.groupby("timestamp")[["actual", "predicted"]]
            .sum()
            .sort_index()
        )

    building_types = [
        ("Dormitory", "Dormitories"),
        ("College Laboratory", "College Laboratories"),
        ("College Classroom", "College Classrooms"),
    ]
  
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

    for ax, (label, title) in zip(axes, building_types):
        agg = agg_by_usage(label)
        ax.plot(
            agg.index,
            agg["actual"],
            label="Actual",
            linewidth=2,
        )
        ax.plot(
            agg.index,
            agg["predicted"],
            label="Predicted",
            linewidth=2,
            alpha=0.9,
        )
        ax.set_title(f"{title}: Actual vs Predicted (October 21–28, 2017)")
        ax.set_ylabel("Electricity, kWh")
        ax.grid(True, alpha=0.3)
        ax.legend()

    axes[-1].set_xlabel("Timestamp")
    plt.tight_layout()
    fig.savefig(FIG_DIR / "week_oct21_28_all_types.png", dpi=300)
    plt.close(fig)

# metrics table (RMSE / MAE / MAPE) + RMSE bar chart
def compute_metrics_by_type(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label in ["Dormitory", "College Laboratory", "College Classroom"]:
        sub = df[df["primary_space_usage"] == label].copy()
        diff = sub["actual"] - sub["predicted"]
        rmse = np.sqrt(np.mean(diff**2))
        mae = np.mean(np.abs(diff))
        mape = (
            np.abs(diff) / sub["actual"].clip(lower=1e-3)
        ).mean() * 100.0

        rows.append(
            {
                "Building Type": label,
                "RMSE": rmse,
                "MAE": mae,
                "MAPE (%)": mape,
            }
        )

    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(FIG_DIR / "metrics_by_building_type.csv", index=False)
    return metrics_df


def plot_rmse_bar(metrics_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(metrics_df["Building Type"], metrics_df["RMSE"])
    ax.set_ylabel("RMSE")
    ax.set_title("Prediction Error by Building Type (2017)")
    ax.grid(True, axis="y", alpha=0.35)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    fig.savefig(FIG_DIR / "rmse_by_building_type.png", dpi=300)
    plt.close(fig)

# feature importance bar plot (log-meter LightGBM model)
def plot_feature_importance() -> None:
    if not MODEL_Y_LOG_PATH.exists():
        raise FileNotFoundError(
            f"{MODEL_Y_LOG_PATH} not found. Run train_and_predict.py first."
        )

    model = lgb.Booster(model_file=str(MODEL_Y_LOG_PATH))

    fig, ax = plt.subplots(figsize=(8, 6))
    lgb.plot_importance(
        model,
        max_num_features=20,
        ax=ax,
        title="Feature Importance – Log Meter Model",
    )
    ax.set_xlabel("Feature importance")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "feature_importance_logmeter.png", dpi=300)
    plt.close(fig)

# monthly seasonal profiles
def plot_seasonal_profiles(df: pd.DataFrame) -> None:
    df = df.copy()
    df["month"] = df["timestamp"].dt.month

    type_definitions = [
        ("Dormitory", "Dormitory", "seasonal_profile_dormitory.png"),
        ("College Laboratory", "College Laboratory", "seasonal_profile_lab.png"),
        ("College Classroom", "College Classroom", "seasonal_profile_classroom.png"),
    ]

    # Separate figures per building type
    for label, title, filename in type_definitions:
        sub = df[df["primary_space_usage"] == label]
        seasonal = (
            sub.groupby("month")[["actual", "predicted"]]
            .mean()
            .reset_index()
        )

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(
            seasonal["month"],
            seasonal["actual"],
            marker="o",
            label="Actual",
        )
        ax.plot(
            seasonal["month"],
            seasonal["predicted"],
            marker="o",
            label="Predicted",
        )
        ax.set_xlabel("Month")
        ax.set_ylabel("Average Electricity")
        ax.set_title(f"Seasonal Profile – {title}")
        ax.legend()
        plt.tight_layout()
        fig.savefig(FIG_DIR / filename, dpi=300)
        plt.close(fig)

    # combined figure
    fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
    for ax, (label, title, _) in zip(axes, type_definitions):
        sub = df[df["primary_space_usage"] == label]
        seasonal = (
            sub.groupby("month")[["actual", "predicted"]]
            .mean()
            .reset_index()
        )
        ax.plot(seasonal["month"], seasonal["actual"], marker="o", label="Actual")
        ax.plot(
            seasonal["month"], seasonal["predicted"], marker="o", label="Predicted"
        )
        ax.set_ylabel("Avg Electricity")
        ax.set_title(f"{title}")
        ax.legend()
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Month")
    plt.tight_layout()
    fig.savefig(FIG_DIR / "seasonal_profile_all_types.png", dpi=300)
    plt.close(fig)
  
# weekly load shapes by hour-of-week
def plot_weekly_shapes(df: pd.DataFrame) -> None:
    df = df.copy()
    df["dow"] = df["timestamp"].dt.dayofweek
    df["hour"] = df["timestamp"].dt.hour
    df["hour_of_week"] = df["dow"] * 24 + df["hour"]

    type_definitions = [
        ("Dormitory", "Dormitory", "weekly_shape_dormitory.png"),
        ("College Laboratory", "College Laboratory", "weekly_shape_lab.png"),
        ("College Classroom", "College Classroom", "weekly_shape_classroom.png"),
    ]

    # separate figures for building type
    for label, title, filename in type_definitions:
        sub = df[df["primary_space_usage"] == label]
        weekly = (
            sub.groupby("hour_of_week")[["actual", "predicted"]]
            .mean()
            .reset_index()
        )

        fig, ax = plt.subplots(figsize=(10, 3))
        ax.plot(weekly["hour_of_week"], weekly["actual"], label="Actual")
        ax.plot(weekly["hour_of_week"], weekly["predicted"], label="Predicted")
        ax.set_xlabel("Hour of Week (0–167)")
        ax.set_ylabel("Average Electricity")
        ax.set_title(f"Typical Weekly Load Shape – {title}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fig.savefig(FIG_DIR / filename, dpi=300)
        plt.close(fig)

    # combined figure 
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    for ax, (label, title, _) in zip(axes, type_definitions):
        sub = df[df["primary_space_usage"] == label]
        weekly = (
            sub.groupby("hour_of_week")[["actual", "predicted"]]
            .mean()
            .reset_index()
        )
        ax.plot(weekly["hour_of_week"], weekly["actual"], label="Actual")
        ax.plot(weekly["hour_of_week"], weekly["predicted"], label="Predicted")
        ax.set_ylabel("Avg Electricity")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Hour of Week (0–167)")
    plt.tight_layout()
    fig.savefig(FIG_DIR / "weekly_shape_all_types.png", dpi=300)
    plt.close(fig)

# full-year 2017 aggregate actual vs predicted
def plot_full_year_2017(df: pd.DataFrame) -> None:
    agg = (
        df.groupby("timestamp")[["actual", "predicted"]]
        .sum()
        .sort_index()
    )
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(
        agg.index,
        agg["actual"],
        label="Actual",
        linewidth=1.5,
    )
    ax.plot(
        agg.index,
        agg["predicted"],
        label="Predicted",
        linewidth=1.5,
        alpha=0.9,
    )
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Electricity, kWh")
    ax.set_title("Aggregate Actual vs Predicted Electricity – 2017")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    fig.savefig(FIG_DIR / "full_year_2017_all_buildings.png", dpi=300)
    plt.close(fig)

# actual vs predicted scatter with correlation line
def plot_actual_vs_pred_scatter(df: pd.DataFrame) -> None:
    sub = df.copy()

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(
        sub["actual"],
        sub["predicted"],
        s=3,
        alpha=0.25,
    )

    lo = min(sub["actual"].min(), sub["predicted"].min())
    hi = max(sub["actual"].max(), sub["predicted"].max())
    ax.plot([lo, hi], [lo, hi], linestyle="--", color="black", linewidth=1)

    ax.set_xlabel("Actual Electricity")
    ax.set_ylabel("Predicted Electricity")
    ax.set_title("Actual vs Predicted Electricity – 2017")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "actual_vs_pred_scatter_2017.png", dpi=300)
    plt.close(fig)


# generate figures
def main():
    df_2017 = load_merged_2017()
    make_week_oct21_28_plot(df_2017)
    metrics_df = compute_metrics_by_type(df_2017)
    plot_rmse_bar(metrics_df)
    plot_feature_importance()
    plot_seasonal_profiles(df_2017)
    plot_weekly_shapes(df_2017)
    plot_full_year_2017(df_2017)
    plot_actual_vs_pred_scatter(df_2017)

    print("Evaluation complete. Figures saved in:", FIG_DIR)

if __name__ == "__main__":
    main()
