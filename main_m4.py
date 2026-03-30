import torch
import pandas as pd
from src.train import train_transformer_model
from src.model import TransformerLikeModel
from dataset.dataset import DatasetTimeSeries, PreprocessingTimeSeries, SheetType
from torch.utils.data import DataLoader
from typing import List, Tuple
import numpy as np
import time
import csv
import os
from pathlib import Path

def parse_dataset_from_csv(
    csv_path: str,
    sheet_type: SheetType,
    output_len: int,
    preprocessing: PreprocessingTimeSeries,
) -> List[Tuple[DatasetTimeSeries, DatasetTimeSeries]]:

    df = pd.read_csv(csv_path)
    datasets = []
    input_len = sheet_type.to_recurrence()

    for _, row in df.iterrows():
        series_id = row["M4id"]
        category = row["category"]

        value_cols = [c for c in df.columns if c.startswith("V") and c != "V1"]
        values = row[value_cols].dropna().astype(float).to_numpy()

        if len(values) < input_len + output_len:
            continue

        train_series = values[:-output_len]
        test_series = values

        train_ds = DatasetTimeSeries(
            series=train_series,
            sheet_type=sheet_type,
            id=series_id,
            category=category,
            output_len=output_len,
            preprocessing=preprocessing,
        )
        test_ds = DatasetTimeSeries(
            series=test_series,
            sheet_type=sheet_type,
            id=series_id,
            category=category,
            output_len=output_len,
            preprocessing=preprocessing,
        )
        datasets.append((train_ds, test_ds))
    return datasets


def run_inference(model, data_loader, device="cpu"):
    """
    Runs model inference on the test DataLoader.
    Batches from DatasetTimeSeries have shape:
    - input:  (batch, timesteps, 1)
    - target: (batch, output_len, 1)
    Returns last-window targets and predictions, shape (output_len,).
    """
    model.eval()
    all_targets = []
    all_preds = []

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs = inputs.to(device)

            preds = model(inputs)

            if preds.dim() == 3:
                preds = preds.squeeze(-1)
            if targets.dim() == 3:
                targets = targets.squeeze(-1)

            all_targets.append(targets.cpu().numpy())
            all_preds.append(preds.cpu().numpy())

    all_targets = np.concatenate(all_targets, axis=0)  # (N_windows, output_len)
    all_preds = np.concatenate(all_preds, axis=0)  # (N_windows, output_len)

    # Ultima finestra = orizzonte reale di forecast
    return all_targets[-1], all_preds[-1]  # (output_len,), (output_len,)


def main():
    torch.manual_seed(42)
    print(
        f"Device: {'CUDA - ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}"
    )

    OUTPUT_LEN = 8
    EMBED_SIZE = 36
    NUM_HEADS = 4
    ENCODER_SIZE = 1
    DECODER_SIZE = 1
    BATCH_SIZE = 16
    EPOCHS = 400
    DROPOUT = 0.05

    SHEET_TYPE = SheetType.QUARTERLY

    datasets = parse_dataset_from_csv(
        "M4sample.csv",
        sheet_type=SHEET_TYPE,
        output_len=OUTPUT_LEN,
        preprocessing=PreprocessingTimeSeries.MIN_MAX,
    )

    input_len = SHEET_TYPE.to_recurrence()

    # CSV summary — una riga per serie
    directory = Path("results")
    log_path = directory / "results_m4_dec.csv"
    if not os.path.exists(log_path):
        with open(log_path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(
                ["id", "category", "train_rmse", "test_rmse", "train_time_s"]
            )

    # CSV detail — una riga per step
    detail_path = directory / "results_m4_dec_detail.csv"
    if not os.path.exists(detail_path):
        with open(detail_path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(
                ["M4id", "category", "step", "ground_truth", "prediction", "residual"]
            )

    for train_dataset, test_dataset in datasets:
        print(f"Training on dataset: {train_dataset.category} (ID: {train_dataset.id})")

        model = TransformerLikeModel(
            embed_size=EMBED_SIZE,
            encoder_size=ENCODER_SIZE,
            decoder_size=DECODER_SIZE,
            output_len=OUTPUT_LEN,
            num_head_enc=NUM_HEADS,
            num_head_dec_1=NUM_HEADS,
            num_head_dec_2=NUM_HEADS,
            dropout=DROPOUT,
            max_seq_length=input_len,
        )

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

        # Training
        start_time = time.time()
        train_loss, test_loss = train_transformer_model(
            model=model,
            epochs=EPOCHS,
            train_data_loader=train_loader,
            test_data_loader=test_loader,
            verbose=False,
            pretrain_seca=True,
            early_stopping=True,
            early_stopping_patience=5,
        )
        end_time = time.time()

        train_rmse = train_loss**0.5
        test_rmse = test_loss**0.5
        tim = end_time - start_time

        print(
            f"ID: {train_dataset.id} - Time: {tim:.2f}s | "
            f"Train RMSE: {train_rmse:.3f} | Test RMSE: {test_rmse:.3f}"
        )

        # Summary CSV
        with open(log_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(
                [
                    train_dataset.id,
                    train_dataset.category,
                    f"{train_rmse:.3f}",
                    f"{test_rmse:.3f}",
                    f"{tim:.2f}",
                ]
            )

        # Inference per il detail CSV
        try:
            last_target, last_pred = run_inference(model, test_loader)
            residuals = last_target - last_pred

            with open(detail_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                for step, (gt, pr, res) in enumerate(
                    zip(last_target, last_pred, residuals), start=1
                ):
                    writer.writerow(
                        [
                            train_dataset.id,
                            train_dataset.category,
                            step,
                            round(float(gt), 4),
                            round(float(pr), 4),
                            round(float(res), 4),
                        ]
                    )
        except Exception as e:
            print(f"  [WARN] Inference fallita per {train_dataset.id}: {e}")


if __name__ == "__main__":
    main()
