# Data Description

This directory should contain the raw data file:

- `ml_capstone_data.xlsx`

Required sheets and columns:

## Sheet: `electricity data`
- `timestamp` (datetime, hourly)
- one column per building, with column names used as `building_id`.

## Sheet: `metadata`
- `building_id` (matching column names in `electricity data`)
- `primary_space_usage` (e.g., `Dormitory`, `College Laboratory`, `College Classroom`)
- `sqft` (floor area in square feet)

## Sheet: `weather` 
- `timestamp`
- `airTemperature`
- `dewTemperature`
- `windSpeed`

This file is included in the repository, but this description is present for reproduciblity.

If the 'ml_capstone_data.xlsx' file is used, then the data_preprocessing.py is not needed before analysis.
