# Given a set of series, the minimalist Transformer-like model is evaluated against the Chronos model and a CNN Attention model.

import torch
import pandas as pd
from typing import List, Tuple
from dataset.dataset import parse_whole_dataset_from_xls, SheetType, PreprocessingTimeSeries, DatasetTimeSeries

from chronos import BaseChronosPipeline
# from tcan.TCAN.model import Transformer
# from tcan.TCAN.utils import Params
# from tcan.TCAN.model.Models import loss_fn

from tqdm import tqdm

def compute_rmse(mu: torch.Tensor, sigma: torch.Tensor, labels: torch.Tensor, *,
                 mask_value: float = 0.0, predictive: bool = False, eps: float = 1e-8) -> float:
    """
    Compute RMSE between mu and labels. If predictive=True, include aleatoric variance:
      RMSE_predictive = sqrt(mean((mu - y)^2 + sigma^2))
    Masks positions where labels == mask_value.
    Returns a Python float (nan if no valid entries).
    """
    # normalize shapes: collapse trailing singleton feature dim
    if labels.dim() == 3 and labels.size(-1) == 1:
        labels = labels.squeeze(-1)
    if mu.dim() == 3 and mu.size(-1) == 1:
        mu = mu.squeeze(-1)
    if sigma is not None and sigma.dim() == 3 and sigma.size(-1) == 1:
        sigma = sigma.squeeze(-1)

    # align time dimension
    min_t = min(mu.size(1), labels.size(1))
    mu = mu[:, :min_t]
    labels = labels[:, :min_t]
    if sigma is not None:
        sigma = sigma[:, :min_t]

    valid = (labels != mask_value)
    if not valid.any():
        return float('nan')

    diff2 = (mu - labels) ** 2
    if predictive:
        if sigma is None:
            raise ValueError("sigma required for predictive RMSE")
        mse = (diff2 + sigma ** 2)[valid].mean()
    else:
        mse = diff2[valid].mean()

    return float(torch.sqrt(mse + eps).item())

def main():
    torch.manual_seed(42)

    CHRONOS = False
    TCAN = True

    OUTPUT_LEN = 18
    INPUT_LEN = 24
    EMBED_SIZE = 36
    NUM_HEADS = 4
    ENCODER_SIZE = 1
    DECODER_SIZE = 1
    BATCH_SIZE = 16
    EPOCHS = 400
    DROPOUT = 0.05

    df = pd.read_csv('results/res_monthly.csv')
    indices = [
        1652, 1546, 1894, 2047, 2255, 2492, 2594, 2658, 2737, 2758, 2817, 2823
    ]

    datasets: List[Tuple[DatasetTimeSeries, DatasetTimeSeries]] = parse_whole_dataset_from_xls("M3C.xls", SheetType.MONTHLY, input_len=INPUT_LEN, output_len=OUTPUT_LEN, preprocessing=PreprocessingTimeSeries.MIN_MAX)
    datasets = [dataset for dataset in datasets if dataset[0].id in indices]

    results = []

    if CHRONOS:
        chronos_model = BaseChronosPipeline.from_pretrained(
            "amazon/chronos-t5-small",
            device_map="cpu",  
            torch_dtype=torch.bfloat16,
        )

        for i, (train_ds, test_ds) in enumerate(tqdm(datasets, desc="Evaluating datasets")):
            input_series = torch.tensor(train_ds.original_series, dtype=torch.float32).unsqueeze(0)  # shape (1, seq_len)

            with torch.no_grad():
                forecast = chronos_model.predict(input_series, prediction_length=len(test_ds.original_series)) # forecast shape: (1, prediction_length)

            forecast = forecast.squeeze(0).cpu().numpy()
            target = test_ds.original_series

            rmse = (((forecast - target) ** 2).mean()) ** 0.5

            results.append({
                "dataset_id": train_ds.id,
                "rmse": rmse,
                "forecast": forecast,
                "target": target,
            })

        print("Chronos Model")
        print("==============")
        for r in results:
            print(f"Dataset {r['dataset_id']}: RMSE={r['rmse']:.4f}")

    # if TCAN:
    #    params = Params('tcan/TCAN/experiments/params_M3C.json')
    #    params.dict["train_window"] = INPUT_LEN
    #    params.dict["test_window"] = OUTPUT_LEN
    #    params.dict["predict_steps"] = 6
    #    params.dict["predict_start"] = OUTPUT_LEN - params.dict["predict_steps"]
    #    params.dict["num_epochs"] = EPOCHS
    #    params.dict["batch_size"] = BATCH_SIZE
#
#    dataset = "M3C"
#
#    tcan_model = Transformer(params, dataset)
#    tcan_model.train()
#
#    optimizer = torch.optim.Adam(tcan_model.parameters(), lr=1e-3)
#    torch.autograd.set_detect_anomaly(True)
#    for train_ds, _ in datasets:
#        train_loader = torch.utils.data.DataLoader(train_ds.tensor_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
#        test_loader = torch.utils.data.DataLoader(train_ds.tensor_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)
#        for _ in range(EPOCHS):
#            if _ % 50 == 0:
#                print(f"Dataset {train_ds.id}: TCAN Epoch {_}")
#            for X_batch, Y_batch in train_loader:
#                X_batch = X_batch.unsqueeze(-1)
#                Y_batch = Y_batch.unsqueeze(-1)
#                optimizer.zero_grad()
#                mu, sigma = tcan_model(X_batch, Y_batch)
#                labels = Y_batch.squeeze(-1)
#                loss = loss_fn(mu, sigma, labels, params.dict["predict_start"])
#                loss.backward()
#                optimizer.step()
#        tcan_model.eval()
#        losses = []
#        for X_batch, Y_batch in test_loader:
#            X_batch = X_batch.unsqueeze(-1)
#            Y_batch = Y_batch.unsqueeze(-1)
#            with torch.no_grad():
#                mu, sigma = tcan_model(X_batch, Y_batch)
#                labels = Y_batch.squeeze(-1)[:, params.dict["predict_start"]:]
#                loss = compute_rmse(mu, sigma, labels, mask_value=0.0, predictive=True)
#                losses.append(loss)
#        print(f"Dataset {train_ds.id}: TCAN Test Loss: {sum(losses)/len(losses):.4f}")
#
#
#
if __name__ == "__main__":
    main()
