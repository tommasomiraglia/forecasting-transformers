# Script per l'utilizzo di timeGpt su M4samples.csv

import time
import pandas as pd
import numpy as np
from nixtla import NixtlaClient
from sklearn.metrics import mean_squared_error
from pathlib import Path

NIXTLA_API_KEY = "nixak-e4e331db7a518fac676737d6eff5f04e9d7e0873a6269b6ab5e375b836f12fb1c03777b18fec5483"

# PARAMETRI STRUTTURALI
MODEL_NAME = "timegpt-1" 
FREQ = "QS"    # frequenza serie: QS=quarterly
LEVEL = [50]  # intervalli di confidenza da calcolare utile per limiti superiori ed inferiori ma non ci serve

# CREAZIONE test e train set gia normalizzati
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

    train = values[:-horizon]
    test = values[-horizon:]

    min_val = train.min()
    max_val = train.max()
    scale = max_val - min_val + 1e-8

    records.append(
        {
            "id": series_id,
            "category": row.get("category", "Unknown"),
            "train": (train - min_val) / scale,
            "test": (test - min_val) / scale,
            "horizon": horizon,
        }
    )

print(f"Serie caricate: {len(records)}\n")

client = NixtlaClient(api_key=NIXTLA_API_KEY)

# timegpt richiede un dataframe con tre colonne obbligatorie:
# - unique_id : identifica la serie 
# - ds        : data/timestamp del valore
# - y         : valore della serie in quel momento
#
train_rows = []
for item in records:
    for t, val in enumerate(item["train"]):
        train_rows.append(
            {
                "unique_id": item["id"],
                "ds": pd.Timestamp("2000-01-01") + pd.DateOffset(months=t * 3),
                "y": val,
            }
        )

df_train = pd.DataFrame(train_rows)

all_ids = []
all_categories = []
all_actuals = []
all_forecasts = []
all_times = []

meta = {item["id"]: item for item in records}

horizon_groups = {}
for item in records:
    horizon_groups.setdefault(item["horizon"], []).append(item["id"])

for horizon, ids in horizon_groups.items():
    df_group = df_train[df_train["unique_id"].isin(ids)].copy()

    start_time = time.time()
    forecast_df = client.forecast(
        df=df_group,
        h=horizon,
        freq=FREQ,
        model=MODEL_NAME,
        level=LEVEL,
        time_col="ds",
        target_col="y",
        id_col="unique_id",
    )
    elapsed = time.time() - start_time
    time_per_series = elapsed / len(ids)

    forecast_col = "TimeGPT"

    for uid in ids:
        fc_vals = forecast_df[forecast_df["unique_id"] == uid][forecast_col].values[
            :horizon
        ]
        item = meta[uid]

        all_ids.append(uid)
        all_categories.append(item["category"])
        all_actuals.append(item["test"])
        all_forecasts.append(fc_vals)
        all_times.append(round(time_per_series, 2))

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

directory = Path("results")
output_path = directory / "timegpt_rmse_resultsM4.csv"
df_results.to_csv(output_path, index=False)
print(f"\nRisultati salvati in: {output_path}")

# CSV dettaglio — una riga per step
rows = []
for uid, cat, actual, forecast in zip(
    all_ids, all_categories, all_actuals, all_forecasts
):
    residuals = actual - forecast
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
detail_path = directory / "timegpt_detail_M4.csv"
df_detail.to_csv(detail_path, index=False)
print(f"Dettaglio step-by-step salvato in: {detail_path}")
