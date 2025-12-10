# Model Checkpoints

Trained LightGBM models are not checked into version control due to size.

To create them:

```bash
python -m src.data_preprocessing
python -m src.train_and_predict
```
This will produce:

```bash
models/lgbm_y_log.txt
models/lgbm_y_log_per_sqft.txt
```

The evaluation script (src.evaluation.py) will load these files
if they exist, or re-train if they are missing.
