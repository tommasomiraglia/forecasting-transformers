
import sys
import os
# Add the project root to sys.path for reliable imports
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from dataset.dataset import DatasetTimeSeries, SheetType, PreprocessingTimeSeries
from src.model import TransformerLikeModel
from src.train import train_transformer_model

import tqdm as tqdm

SEED = 42
OUTPUT_LEN = 18
INPUT_LEN = 24
BATCH_SIZE = 32
TRAIN_SERIES_LENGTH = 1000
TEST_SERIES_LENGTH = 320
SEASONALITY_PERIOD = 4
TRAIN_AMPLITUDE = 1.0
NOISE_LEVEL = 0.25
SEASONAL_PERIOD_LEVELS = [4, 8, 12, 16, 20, 24, 27, 30, 32, 36]
SERIES_PER_LEVEL = 10
EPOCHS = 10
TREND = 0.001

def create_seasonal_data(length, seasonality_period, amplitude=1.0, noise_level=1.0, trend=0.001):
    """
    Create synthetic seasonal data.

    Parameters:
    - length: Total length of the time series.
    - seasonality_period: The period of the seasonality (e.g., 12 for monthly data with yearly seasonality).
    - amplitude: The amplitude of the seasonal component.
    - noise_level: The standard deviation of the Gaussian noise added to the data.
    - trend: The trend component of the time series.
    Returns:
    - A numpy array containing the synthetic seasonal data.
    """
    t = np.arange(length)
    trend_component = trend * t
    seasonal_component = amplitude * (
        np.sin(2 * np.pi * t / seasonality_period) +
        0.5 * np.sin(4 * np.pi * t / seasonality_period) +
        0.25 * np.sin(6 * np.pi * t / seasonality_period)
    )
    noise = np.random.normal(0, noise_level, size=length)
    
    return seasonal_component + noise + trend_component


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_dataset(series: np.ndarray, identifier: int) -> DatasetTimeSeries:
    return DatasetTimeSeries(
        series=series,
        sheet_type=SheetType.OTHER,
        id=identifier,
        category="synthetic-seasonality",
        output_len=OUTPUT_LEN,
        preprocessing=PreprocessingTimeSeries.MIN_MAX,
    )


def evaluate_series_rmse(model: TransformerLikeModel, dataset: DatasetTimeSeries) -> float:
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    criterion = torch.nn.MSELoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    losses = []
    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            y_pred = model(x_batch)
            losses.append(criterion(y_pred, y_batch).item())

    if not losses:
        raise RuntimeError("Evaluation dataset produced no batches. Increase series length.")
    return float(np.sqrt(np.mean(losses)))


def main() -> None:
    set_seed(SEED)

    train_series = create_seasonal_data(
        length=TRAIN_SERIES_LENGTH,
        seasonality_period=SEASONALITY_PERIOD,
        amplitude=TRAIN_AMPLITUDE,
        noise_level=NOISE_LEVEL,
        trend=TREND
    ).astype(np.float32)
    train_dataset = build_dataset(train_series, identifier=0)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = TransformerLikeModel(
        embed_size=24,
        encoder_size=2,
        decoder_size=2,
        output_len=OUTPUT_LEN,
        num_head_enc=4,
        num_head_dec_1=4,
        num_head_dec_2=4,
        dropout=0.10,
        max_seq_length=INPUT_LEN,
    )

    train_rmse_sq, _ = train_transformer_model(
        model=model,
        epochs=EPOCHS,
        train_data_loader=train_loader,
        test_data_loader=train_loader,
        verbose=True,
        pretrain_seca=True,
        early_stopping=True,
        early_stopping_patience=6,
        learning_rate=2e-3,
    )
    print(f"Train RMSE: {train_rmse_sq ** 0.5:.4f}")

    seasonal_period_levels = SEASONAL_PERIOD_LEVELS
    seasonal_results = []

    for level_idx, seasonal_period in tqdm.tqdm(enumerate(seasonal_period_levels), total=len(seasonal_period_levels)):
        for sample_idx in range(SERIES_PER_LEVEL):
            test_series = create_seasonal_data(
                length=TEST_SERIES_LENGTH,
                seasonality_period=int(seasonal_period),
                amplitude=TRAIN_AMPLITUDE,
                noise_level=NOISE_LEVEL,
                trend=TREND
            ).astype(np.float32)
            test_dataset = build_dataset(test_series, identifier=1000 + level_idx * SERIES_PER_LEVEL + sample_idx)
            rmse = evaluate_series_rmse(model, test_dataset)
            seasonal_results.append(
                {
                    "seasonal_period": int(seasonal_period),
                    "series_idx": sample_idx,
                    "rmse": rmse,
                }
            )
    
    trend_period_levels = [0.0, 0.0001, 0.0002, 0.0005, 0.001, 0.0015, 0.002, 0.005, 0.01]
    trend_results = []

    for level_idx, trend in tqdm.tqdm(enumerate(trend_period_levels), total=len(trend_period_levels)):
        for sample_idx in range(SERIES_PER_LEVEL):
            test_series = create_seasonal_data(
                length=TEST_SERIES_LENGTH,
                seasonality_period=SEASONALITY_PERIOD,
                amplitude=TRAIN_AMPLITUDE,
                noise_level=NOISE_LEVEL,
                trend=trend
            ).astype(np.float32)
            test_dataset = build_dataset(test_series, identifier=2000 + level_idx * SERIES_PER_LEVEL + sample_idx)
            rmse = evaluate_series_rmse(model, test_dataset)
            trend_results.append(
                {
                    "trend": trend,
                    "series_idx": sample_idx,
                    "rmse": rmse,
                }
            )

    seasonal_df = pd.DataFrame(seasonal_results)
    grouped = (
        seasonal_df.groupby("seasonal_period", as_index=False)
        .agg(
            rmse_mean=("rmse", "mean"),
            rmse_std=("rmse", "std"),
            rmse_min=("rmse", "min"),
            rmse_max=("rmse", "max"),
            n_series=("rmse", "count"),
        )
        .sort_values("seasonal_period")
    )

    best_row = grouped.loc[grouped["rmse_mean"].idxmin()]
    look_row = grouped[grouped["seasonal_period"] == SEASONALITY_PERIOD]
    if not look_row.empty:
        rmse_24 = float(look_row.iloc[0]["rmse_mean"])
        print(
            f"\nPeriod {SEASONALITY_PERIOD} mean RMSE: {rmse_24:.6f} | "
            f"Best period: {int(best_row['seasonal_period'])} (RMSE {float(best_row['rmse_mean']):.6f})"
        )

    print("\nGrouped results by seasonal period:")
    print(grouped.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    trend_df = pd.DataFrame(trend_results)
    grouped = (
        trend_df.groupby("trend", as_index=False)
        .agg(
            rmse_mean=("rmse", "mean"),
            rmse_std=("rmse", "std"),
            rmse_min=("rmse", "min"),
            rmse_max=("rmse", "max"),
            n_series=("rmse", "count"),
        ).sort_values("trend")
    )

    best_row = grouped.loc[grouped["rmse_mean"].idxmin()]
    look_row = grouped[grouped["trend"] == TREND]
    if not look_row.empty:
        rmse_trend = float(look_row.iloc[0]["rmse_mean"])
        print(
            f"\nTrend {TREND} mean RMSE: {rmse_trend:.6f} | "
            f"Best trend: {best_row['trend']} (RMSE {float(best_row['rmse_mean']):.6f})"
        )

    print("\nGrouped results by trend:")
    print(grouped.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    output_path = "results/seasonality_grouped_results.csv"
    grouped.to_csv(output_path, index=False)
    print(f"\nSaved grouped results to: {output_path}")


if __name__ == "__main__":
    main()