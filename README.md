# Machine Learning for Building Energy Consumption Prediction and Comparison

Author: Olivia Solomon  

This repository contains the code for my ML capstone project on forecasting hourly
electricity consumption for campus-type buildings using a LightGBM model inspired
by the ASHRAE Great Energy Predictor III (GEPIII) competition.

The project trains on 2016 meter data and predicts 2017 hourly electricity for
three primary building types:

- Dormitory
- College Laboratory
- College Classroom

The model uses weather, calendar features, holidays, and building metadata
(primary usage, floor area) and evaluates performance with RMSE, MAE, and MAPE.

---

---

## 🔧 Quick Start

### 1. Create and activate a Python environment (Python 3.10+ recommended)

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```
### 2. Add the dataset

Place the Excel file into the data/ directory:
```bash
data/ml_capstone_data.xlsx
```
Required sheets:
- electricity data: hourly readings per building
- metadata: building_id, primary_space_usage, sqft
- weather: airTemperature, dewTemperature, windSpeed

### 3. Run preprocessing and model training

Note: if using the provided ml_capstone_data.xlsx, preprocessing is not required.
```bash
python -m src.data_preprocessing
python -m src.train_and_predict
```
This will generate:
- data/electricity_long_2017.parquet
- outputs/electricity_2017_predictions_long.csv
- trained LightGBM model files in the models/ directory

### 4. Generate figures and evaluation metrics
```bash
python -m src.evaluation
```
All plots will appear within the figures/ directory.

### 5. Model Checkpoints
To regenerate model files:
```bash
python -m src.train_and_predict
```
This will create:
```bash
models/lgbm_y_long.txt
models/lgbm_y_log_per_sqft.txt
```
Which will be used by:
```bash
src/evaluation.py
```

## Repository Structure

```text
.
├── README.md
├── requirements.txt
├── data
│   ├── ml_capstone_data.xlsx
|   ├── electricity_long_2017.parquet  
│   └── README_DATA.md
├── src
│   ├── data_preprocessing.py
│   ├── train_and_predict.py
│   └── evaluation.py
├── outputs
│   ├── electricity_2017_predictions_long.csv
├── notebooks
│   └── 01_sample_analysis.ipynb
├── models
│   └── README_MODELS.md
│   └── lgbm_y_log.md
│   └── lgbm_y_log_per_sqft.md
└── figures
  ├── week_oct21_28_all_types.png
  ├── rmse_by_building_type.png
  ├── feature_importance_logmeter.png
  ├── seasonal_profile_dormitory.png
  ├── seasonal_profile_lab.png
  ├── seasonal_profile_classroom.png
  ├── seasonal_profile_all_types.png
  ├── weekly_shape_dormitory.png
  ├── weekly_shape_lab.png
  ├── weekly_shape_classroom.png
  ├── weekly_shape_all_types.png
  ├── full_year_2017_all_buildings.png
  └── actual_vs_pred_scatter_2017.png
```
### High-Level Methodology
Feature engineering:
- temporal features, weather lags, rolling means, floor-area normalization
- sinusoidal encodings for hour-of-day, day-of-week, day-of-year
Model:
- LightGBM regression
- Two-model ensemble (log-meter + log_intensity)
Training:
- Full year 2016 meter data
- Final 30 days of 2016 for early stopping
Prediction:
- Full hourly forecast for 2017 in kWh

### References
- [BDG2 dataset](https://github.com/buds-lab/building-data-genome-project-2)
- [GEPIII results](https://www.kaggle.com/competitions/ashrae-energy-prediction/leaderboard)
- [GEPIII 1st place solution writeup](https://www.kaggle.com/competitions/ashrae-energy-prediction/writeups/isamu-matt-1st-place-solution-team-isamu-matt)
  

