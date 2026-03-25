import time
import pandas as pd
import numpy as np
import torch
from chronos import ChronosPipeline
from sklearn.metrics import mean_squared_error

torch.manual_seed(42)
df = pd.read_csv("M4sample.csv")
value_cols = [c for c in df.columns if c.startswith("V") and c != "V1"]

records = []
for _, row in df.iterrows():
    series_id = row["M4id"]
    horizon = int(row["Horizon"])
    values = (
        pd.to_numeric(row[value_cols], errors="coerce").dropna().values.astype(float)
    )
    if len(values) <= horizon:
        print(
            f"  [SKIP] {series_id}: troppo corta ({len(values)} valori, horizon={horizon})"
        )
        continue
    records.append(
        {
            "id": series_id,
            "category": row.get("category", "Unknown"),
            "train": values[:-horizon],
            "test": values[-horizon:],
            "horizon": horizon,
        }
    )

MODEL_NAME = "amazon/chronos-t5-small"
DEVICE = "cpu"
NUM_SAMPLES = 20

pipeline = ChronosPipeline.from_pretrained(
    MODEL_NAME,
    device_map=DEVICE,
    torch_dtype=torch.bfloat16,
)

all_ids = []
all_categories = []
all_actuals = []
all_forecasts = []
all_residuals = []
all_times = []

for item in records:
    train = item["train"]
    test = item["test"]

    min_val = train.min()
    max_val = train.max()
    scale = max_val - min_val + 1e-8

    train_norm = (train - min_val) / scale
    test_norm = (test - min_val) / scale

    context = torch.tensor(train_norm, dtype=torch.float32).unsqueeze(0)

    start_time = time.time()
    forecast = pipeline.predict(
        context,
        prediction_length=item["horizon"],
        num_samples=NUM_SAMPLES,
    )
    tim = time.time() - start_time

    median_forecast = forecast[0].median(dim=0).values.numpy()
    residuals = test_norm - median_forecast

    all_ids.append(item["id"])
    all_categories.append(item["category"])
    all_actuals.append(test_norm)
    all_forecasts.append(median_forecast)
    all_residuals.append(residuals)
    all_times.append(tim)

print(f"Previsioni completate per {len(all_forecasts)} serie.\n")


def compute_rmse(actual, forecast):
    return np.sqrt(mean_squared_error(actual, forecast))


rmse_list = []
for actual, forecast in zip(all_actuals, all_forecasts):
    rmse_list.append(compute_rmse(actual, forecast))

print(f"RMSE medio globale (normalizzato): {np.nanmean(rmse_list):.4f}")

# CSV principale — una riga per serie
df_results = pd.DataFrame(
    {
        "M4id": all_ids,
        "category": all_categories,
        "rmse": rmse_list,
        "time": all_times,
    }
)
df_results["time"] = df_results["time"].round(2)
df_results["rmse"] = df_results["rmse"].round(3)

output_path = "chronos_nrmse_resultsM4.csv"
df_results.to_csv(output_path, index=False)
print(f"Risultati salvati in: {output_path}")

# CSV dettaglio — una riga per step
rows = []
for uid, cat, actual, forecast, residuals in zip(
    all_ids, all_categories, all_actuals, all_forecasts, all_residuals
):
    for step, (a, f, r) in enumerate(zip(actual, forecast, residuals), start=1):
        rows.append(
            {
                "M4id": uid,
                "category": cat,
                "step": step,
                "ground_truth": round(float(a), 4),
                "prediction": round(float(f), 4),
                "residual": round(float(r), 4),
            }
        )

df_detail = pd.DataFrame(rows)
detail_path = "chronos_detail_M4.csv"
df_detail.to_csv(detail_path, index=False)
print(f"Dettaglio step-by-step salvato in: {detail_path}")
