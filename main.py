import torch
import pandas as pd
from src.seca import ScalarExpansionContractiveAutoencoder
from src.train import train_transformer_model
from src.model import TransformerLikeModel
from dataset.dataset import (
    parse_whole_dataset_from_xls,
    SheetType,
    PreprocessingTimeSeries,
    DatasetTimeSeries,
)

from torch.utils.data import DataLoader
from sklearn.ensemble import RandomForestRegressor
from typing import List, Tuple, Dict
import numpy as np
import pickle
import random
import time


class Result:

    def __init__(self, num_models: int):
        self.num_models = num_models
        self.train_loss = [float("inf") for _ in range(num_models)]
        self.test_loss = [float("inf") for _ in range(num_models)]
        self.predictions: List[List[float]] = [[] for _ in range(num_models)]

    def __getitem__(self, idx: int) -> Tuple[float, float]:
        return self.train_loss[idx], self.test_loss[idx]

    def __setitem__(self, idx: int, value: Tuple[float, float]):
        self.train_loss[idx], self.test_loss[idx] = value

    def get_predictions(self, idx: int) -> List[float]:
        return self.predictions[idx]

    def set_predictions(self, idx: int, preds: List[float]):
        self.predictions[idx] = preds


def main():
    torch.manual_seed(42)

    OUTPUT_LEN = 18
    INPUT_LEN = 24
    EMBED_SIZE = 36
    NUM_HEADS = 4
    ENCODER_SIZE = 1
    DECODER_SIZE = 1
    BATCH_SIZE = 16
    EPOCHS = 400
    DROPOUT = 0.05

    df = pd.read_csv("results/res_monthly.csv")
    # indices = df.id.tolist()
    indices = [1652, 1546, 1894, 2047, 2255, 2492, 2594, 2658, 2737, 2758, 2817, 2823]
    tim = 0

    datasets: List[Tuple[DatasetTimeSeries, DatasetTimeSeries]] = (
        parse_whole_dataset_from_xls(
            "M3C.xls",
            SheetType.MONTHLY,
            input_len=INPUT_LEN,
            output_len=OUTPUT_LEN,
            preprocessing=PreprocessingTimeSeries.MIN_MAX,
        )
    )
    datasets = [dataset for dataset in datasets if dataset[0].id in indices]
    results: List[Result] = []

    for train_dataset, test_dataset in datasets:
        results.append(Result(num_models=2))

        print(f"Training on dataset: {train_dataset.category} (ID: {train_dataset.id})")
        # print(f"Number of training samples: {len(train_dataset)}, Number of testing samples: {len(test_dataset)}")

        model: TransformerLikeModel = TransformerLikeModel(
            embed_size=EMBED_SIZE,
            encoder_size=ENCODER_SIZE,
            decoder_size=DECODER_SIZE,
            output_len=OUTPUT_LEN,
            num_head_enc=NUM_HEADS,
            num_head_dec_1=NUM_HEADS,
            num_head_dec_2=NUM_HEADS,
            dropout=DROPOUT,
            max_seq_length=INPUT_LEN,
        )
        # print("Number of trainable parameters: ", sum(p.numel() for p in model.parameters() if p.requires_grad))

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

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
        tim = end_time - start_time
        print(
            f"ID: {train_dataset.id} - Time for training Transformer: {tim} seconds"
        )

        results[-1][0] = (train_loss**0.5, test_loss**0.5)  # RMSE
        all_preds = []
        ######################################
        for i, (X_batch, _) in enumerate(test_loader):
            preds = model(X_batch)
            p_np = preds.detach().cpu().numpy()
            all_preds.append(p_np)
        all_preds = np.concatenate(all_preds, axis=0)
        _, y_test_np = test_dataset.np_datasets
        ######################################

        clf = RandomForestRegressor(n_estimators=250, random_state=42)
        X_np, y_np = train_dataset.np_datasets
        start_time = time.time()
        clf.fit(X_np, y_np)
        end_time = time.time()
        tim = end_time - start_time
        print(f"ID: {train_dataset.id} - Time for training Forest: {tim :.2f} seconds")

        ######################################

        y_p_train = clf.predict(X_np)
        train_rmse = np.sqrt(np.mean((y_p_train - y_np) ** 2))  # RMSE
        X_np, y_np = test_dataset.np_datasets
        y_p_test = clf.predict(X_np)
        test_rmse = np.sqrt(np.mean((y_p_test - y_np) ** 2))  # RMSE
        results[-1][1] = (train_rmse, test_rmse)
        results[-1].set_predictions(1, y_p_test.flatten().tolist())

        with open("results_log.txt", "a", encoding="utf-8") as logf:
            logf.write(f"Dataset: {train_dataset.category} (ID: {train_dataset.id})\n")
            logf.write(
                f"Transformer - Train RMSE: {results[-1][0][0]:.4f}, Test RMSE: {results[-1][0][1]:.4f}\n"
            )
            logf.write(
                f"Random Forest - Train RMSE: {results[-1][1][0]:.4f}, Test RMSE: {results[-1][1][1]:.4f}\n"
            )
            logf.write("Transformer predictions (shape {}):\n".format(all_preds.shape))
            logf.write(
                np.array2string(
                    all_preds,
                    precision=4,
                    separator=", ",
                    max_line_width=2000,
                    threshold=1000000,
                )
                + "\n"
            )
            logf.write("Forest predictions (shape {}):\n".format(y_p_test.shape))
            logf.write(
                np.array2string(
                    y_p_test,
                    precision=4,
                    separator=", ",
                    max_line_width=2000,
                    threshold=1000000,
                )
                + "\n"
            )
            logf.write(
                "Ground-truth test targets (shape {}):\n".format(y_test_np.shape)
            )
            logf.write(
                np.array2string(
                    y_test_np,
                    precision=4,
                    separator=", ",
                    max_line_width=2000,
                    threshold=1000000,
                )
                + "\n"
            )
            logf.write("\n")


if __name__ == "__main__":
    main()
