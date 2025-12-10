import os
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
OUTPUT_PATH = DATA_DIR / "electricity_long_2017.parquet"


def load_raw_data():
    xlsx_path = DATA_DIR / "ml_capstone_data.xlsx"
    if not xlsx_path.exists():
        raise FileNotFoundError(f"Expected {xlsx_path} to exist")

    elec = pd.read_excel(
        xlsx_path,
        sheet_name="electricity data",
        parse_dates=["timestamp"],
    )
    meta = pd.read_excel(xlsx_path, sheet_name="metadata")
    weather = pd.read_excel(
        xlsx_path,
        sheet_name="weather",
        parse_dates=["timestamp"],
    )

    return elec, meta, weather

def preprocess():
    elec, meta, weather = load_raw_data()

    # converts from wide to long
    elec_long = elec.melt(
        id_vars=["timestamp"],
        var_name="building_id",
        value_name="meter_reading",
    )

    # filters to 2017 only because training uses 2016, model script will re-load
    elec_long = elec_long[elec_long["timestamp"].dt.year.isin([2016, 2017])]

    # clean IDs
    elec_long["building_id"] = elec_long["building_id"].astype(str).str.strip()
    meta["building_id"] = meta["building_id"].astype(str).str.strip()

    # merge metadata
    df = elec_long.merge(meta, on="building_id", how="left")

    # merge weather data
    df = df.merge(weather, on="timestamp", how="left")

    # drop bad readings
    df = df[df["meter_reading"].notna()]
    df = df[df["meter_reading"] >= 0]

    # save processed data
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_PATH, index=False)
    print(f"Saved processed long-format data to {OUTPUT_PATH}")


if __name__ == "__main__":
    preprocess()
