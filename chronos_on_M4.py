import time
import pandas as pd
import numpy as np
import torch
from chronos import ChronosPipeline
from sklearn.metrics import mean_squared_error

torch.manual_seed(42)
df = pd.read_csv("M4sample.csv")
value_cols = [c for c in df.columns if c.startswith("V") and c != "V1"]

# preparazione train/test set

records = []

for _, row in df.iterrows():
    series_id = row["M4id"]
    horizon = int(row["Horizon"])

    # Estrai valori numerici a partire da V2, scarta NaN
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
NUM_SAMPLES = 20  # quante previsioni vuoi fare così da avere una distribuzione di possibili futuri


pipeline = ChronosPipeline.from_pretrained(
    MODEL_NAME,
    device_map=DEVICE,
    torch_dtype=torch.bfloat16,
)


all_ids = []
all_categories = []
all_actuals = []
all_forecasts = []
all_times = []

for i, item in enumerate(records):
    context = torch.tensor(item["train"], dtype=torch.float32).unsqueeze(0)  
    # unsqueeze ho un solo esempio, ma trattalo comunque come una batch se ad es.
    # vuoi passare più array per il train
    start_time = time.time()
    forecast = pipeline.predict(
        context,
        prediction_length=item["horizon"],
        num_samples=NUM_SAMPLES,
    )
    end_time = time.time()
    tim = end_time - start_time
    # mediana sulle previsione
    median_forecast = forecast[0].median(dim=0).values.numpy()

    all_ids.append(item["id"])
    all_categories.append(item["category"])
    all_actuals.append(item["test"])
    all_forecasts.append(median_forecast)
    all_times.append(tim)
print(f"Previsioni completate per {len(all_forecasts)} serie.\n")


# 5. CALCOLO NRMSE
# NRMSE = RMSE / mean(|actual_test|)
def compute_nrmse(actual, forecast):
    rmse = np.sqrt(mean_squared_error(actual, forecast))
    mean_actual = np.mean(np.abs(actual))
    return rmse / mean_actual if mean_actual > 0 else np.nan


nrmse_list = []
for actual, forecast in zip(all_actuals, all_forecasts):
    nrmse = compute_nrmse(actual, forecast)
    nrmse_list.append(nrmse)

print(f"NRMSE medio globale: {np.nanmean(nrmse_list):.4f}")


# 6. NRMSE PER CATEGORIA
df_results = pd.DataFrame(
    {
        "M4id": all_ids,
        "category": all_categories,
        "nrmse": nrmse_list,
        "time": all_times,
    }
)
df_results["time"] = df_results["time"].round(2)
df_results["nrmse"] = df_results["nrmse"].round(3)

# print("NRMSE medio per categoria:")
# print(df_results.groupby("category")["nrmse"].mean().round(4).to_string())

output_path = "chronos_nrmse_resultsM4.csv"
df_results.to_csv(output_path, index=False)
print(f"\nRisultati salvati in: {output_path}")
