import time
import pandas as pd
import numpy as np
from nixtla import NixtlaClient
from sklearn.metrics import mean_squared_error

NIXTLA_API_KEY = "nixak-e4e331db7a518fac676737d6eff5f04e9d7e0873a6269b6ab5e375b836f12fb1c03777b18fec5483"  

np.random.seed(42)

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

print(f"Serie caricate: {len(records)}\n")

client = NixtlaClient(api_key=NIXTLA_API_KEY)
# TimeGPT lavora con un DataFrame "lungo" con colonne: unique_id, ds, y
train_rows = []
for item in records:
    n = len(item["train"])
    for t, val in enumerate(item["train"]):
        train_rows.append(
            {
                "unique_id": item["id"],
                "ds": pd.Timestamp("2000-01-01") + pd.DateOffset(months=t),
                "y": val,
            }
        )

df_train = pd.DataFrame(train_rows)

# PREVISIONI CON TIMEGPT
# TimeGPT supporta forecast batch: tutte le serie in una sola chiamata API.
# Raggruppiamo per horizon, dato che M4 ha orizzonti diversi per categoria.

all_ids = []
all_categories = []
all_actuals = []
all_forecasts = []
all_times = []

meta = {item["id"]: item for item in records}

horizon_groups = {}
for item in records:
    h = item["horizon"]
    horizon_groups.setdefault(h, []).append(item["id"])

for horizon, ids in horizon_groups.items():
    df_group = df_train[df_train["unique_id"].isin(ids)].copy()

    start_time = time.time()
    forecast_df = client.forecast(
        df=df_group,
        h=horizon,
        freq="MS",  # Monthly Start – adatta se usi altra frequenza
        model="timegpt-1",  # oppure "timegpt-1-long-horizon" per h > 12
        level=[50],  # intervallo al 50% → mediana
        time_col="ds",
        target_col="y",
        id_col="unique_id",
    )
    elapsed = time.time() - start_time

    time_per_series = elapsed / len(ids)

    # TimeGPT restituisce la mediana nella colonna "TimeGPT"
    # (o "TimeGPT-median" se richiedi livelli di confidenza)
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


# CALCOLO NRMSE 
def compute_nrmse(actual, forecast):
    rmse = np.sqrt(mean_squared_error(actual, forecast))
    mean_actual = np.mean(np.abs(actual))
    return rmse / mean_actual if mean_actual > 0 else np.nan


nrmse_list = []
for actual, forecast in zip(all_actuals, all_forecasts):
    nrmse = compute_nrmse(actual, forecast)
    nrmse_list.append(nrmse)

print(f"NRMSE medio globale: {np.nanmean(nrmse_list):.4f}")

# NRMSE PER CATEGORIA 
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

output_path = "timegpt_nrmse_resultsM4.csv"
df_results.to_csv(output_path, index=False)
print(f"\nRisultati salvati in: {output_path}")
