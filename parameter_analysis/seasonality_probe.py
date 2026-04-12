"""
seasonality_probe.py
--------------------
Linear-probing experiment: can a linear classifier recover the seasonality
period of a time series from the frozen activations of the first encoder layer?
"""

import sys, os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

from dataset.dataset import DatasetTimeSeries, SheetType, PreprocessingTimeSeries
from src.model import TransformerLikeModel
from src.train import train_transformer_model

# --- Config ---
SEED = 42
OUTPUT_LEN = 18
INPUT_LEN = 24
BATCH_SIZE = 32
TRAIN_SEASONALITY = 4
TRAIN_SERIES_LEN = 1000
AMPLITUDE = 1.0
NOISE = 0.25
TREND = 0.001
EPOCHS = 10

PROBE_SEASONALITIES = [4, 8, 12, 16, 20, 24, 27, 30, 32, 36]
PROBE_SERIES_PER_CLASS = 40
PROBE_SERIES_LEN = 320
TEST_SIZE = 0.20


def create_seasonal_data(length, period, amplitude=1.0, noise=0.25, trend=0.001):
    t = np.arange(length)
    s = amplitude * (
        np.sin(2 * np.pi * t / period)
        + 0.5 * np.sin(4 * np.pi * t / period)
        + 0.25 * np.sin(6 * np.pi * t / period)
    )
    return (s + np.random.normal(0, noise, length) + trend * t).astype(np.float32)


def build_dataset(series, identifier):
    return DatasetTimeSeries(
        series=series,
        sheet_type=SheetType.OTHER,
        id=identifier,
        category="probe",
        output_len=OUTPUT_LEN,
        preprocessing=PreprocessingTimeSeries.MIN_MAX,
    )


def collect_activations(model, hook_layer, seasonalities, n_series, series_len):
    """Run probe series through the frozen model; return (X, y)."""
    device = next(model.parameters()).device
    activations = []

    def hook_fn(_, __, output):
        t = output[0] if isinstance(output, tuple) else output
        pooled = t.mean(dim=1) if t.dim() == 3 else t
        activations.append(pooled.detach().cpu().numpy())

    handle = hook_layer.register_forward_hook(hook_fn)
    all_X, all_y = [], []

    model.eval()
    with torch.no_grad():
        for class_idx, period in enumerate(seasonalities):
            activations.clear()
            for i in range(n_series):
                series = create_seasonal_data(
                    series_len, period, AMPLITUDE, NOISE, TREND
                )
                loader = DataLoader(
                    build_dataset(series, class_idx * 1000 + i),
                    batch_size=BATCH_SIZE,
                    shuffle=False,
                )
                for x, _ in loader:
                    model(x.to(device))
            X_cls = np.concatenate(activations)
            all_X.append(X_cls)
            all_y.append(np.full(len(X_cls), class_idx, dtype=np.int64))
            print(f"  period={period:>3}  windows={len(X_cls)}")

    handle.remove()
    return np.concatenate(all_X), np.concatenate(all_y)


def main():
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Train transformer
    print(f"=== Step 1: Training on seasonality={TRAIN_SEASONALITY} ===")
    train_series = create_seasonal_data(TRAIN_SERIES_LEN, TRAIN_SEASONALITY)
    train_loader = DataLoader(
        build_dataset(train_series, 0), batch_size=BATCH_SIZE, shuffle=True
    )

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
    ).to(device)

    rmse_sq, _ = train_transformer_model(
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
    print(f"Train RMSE: {rmse_sq ** 0.5:.4f}\n")

    for p in model.parameters():
        p.requires_grad_(False)

    # 2. Hook on first encoder layer
    first_layer = model.encoder[0]
    print(f"[probe] Hook on: encoder[0]  ({type(first_layer).__name__})\n")

    # 3. Collect activations
    print("=== Step 2: Collecting activations ===")
    X, y = collect_activations(
        model,
        first_layer,
        PROBE_SEASONALITIES,
        PROBE_SERIES_PER_CLASS,
        PROBE_SERIES_LEN,
    )
    print(f"\nSamples: {len(X)}  |  dim: {X.shape[1]}\n")

    # 4. Train linear probe
    print("=== Step 3: Linear probe ===")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    probe = LogisticRegression(max_iter=1000, solver="lbfgs", C=1.0, random_state=SEED)
    probe.fit(X_train, y_train)
    y_pred = probe.predict(X_test)

    # 5. Results
    class_names = [f"s={p}" for p in PROBE_SEASONALITIES]
    print(f"\nOverall accuracy: {accuracy_score(y_test, y_pred):.4f}\n")
    print(classification_report(y_test, y_pred, target_names=class_names, digits=3))

    cm = confusion_matrix(y_test, y_pred)
    per_class_acc = cm.diagonal() / cm.sum(axis=1)
    summary = pd.DataFrame(
        {
            "seasonality": PROBE_SEASONALITIES,
            "accuracy": per_class_acc,
            "n_correct": cm.diagonal(),
            "n_total": cm.sum(axis=1),
        }
    )
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    os.makedirs("results", exist_ok=True)
    summary.to_csv("results/seasonality_probe_results.csv", index=False)


if __name__ == "__main__":
    main()
