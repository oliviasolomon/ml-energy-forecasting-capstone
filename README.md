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

## Repository Structure

```text
.
├── README.md
├── requirements.txt
├── data
│   ├── ml_capstone_data.xlsx      
│   └── README_DATA.md
├── src
│   ├── data_preprocessing.py
│   ├── train_and_predict.py
│   └── evaluation.py
├── notebooks
│   └── 01_example_analysis.ipynb
├── models
│   └── README_MODELS.md
└── figures
    ├── week_oct21_28_all_types.png
    ├── rmse_by_building_type.png
    ├── feature_importance_logmeter.png
    ├── seasonal_profile_lab.png
    └── weekly_shape_classroom.png
