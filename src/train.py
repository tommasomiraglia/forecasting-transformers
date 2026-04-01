import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np

from src.model import TransformerLikeModel, EncoderOnlyModel
from src.seca import ScalarExpansionContractiveAutoencoder as Seca
from torch.utils.data import DataLoader

from typing import Tuple, Union

LEARNING_RATE = 1e-4

# Modificato per ottenere gli array con le previsioni finali
def train_transformer_model(
    model: TransformerLikeModel,
    epochs: int,
    train_data_loader: DataLoader,
    test_data_loader: DataLoader,
    verbose: bool = True,
    teacher_forcing_ratio: float = 1.0,
    pretrain_seca: bool = True,
    check_losses: bool = False,
    delta: bool = False,
    early_stopping: bool = False,
    early_stopping_patience: int = 10,
    early_stopping_min_delta: float = 1e-4,
    learning_rate: float = LEARNING_RATE,
    return_preds: bool = False,
) -> Union[Tuple[float, float], Tuple[float, float, np.ndarray, np.ndarray]]:

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    assert (
        0.0 <= teacher_forcing_ratio <= 1.0
    ), "teacher_forcing_ratio must be in [0, 1]"
    assert (
        getattr(model, "output_len", None) is not None and model.output_len > 0
    ), "model.output_len must be a positive integer"

    model.train()
    if pretrain_seca and model.seca is not None:
        model.seca.start()

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
    criterion = nn.MSELoss()

    ret_train_loss = 0.0
    ret_test_loss = 0.0

    for epoch in range(epochs):
        epoch_loss = 0

        if early_stopping and epoch == 0:
            best_val_loss = float("inf")
            epochs_no_improve = 0

        for X_batch, y_batch in train_data_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            batch_size, output_len, _ = y_batch.shape

            optimizer.zero_grad()
            Y = model.cls_token.expand((batch_size, 1, -1))
            total_loss = torch.tensor(0.0, device=device)
            preds = []

            for step in range(model.output_len):
                output = model.single_forward((X_batch, Y))
                if random.random() < teacher_forcing_ratio:
                    to_append = model.seca.encode(y_batch[:, step].unsqueeze(1))
                else:
                    to_append = output.unsqueeze(1)
                Y = torch.cat([Y, to_append], dim=1)
                y = model.seca.decode(output)
                if delta:
                    y += X_batch[:, -1, :] if step == 0 else preds[-1]
                    preds.append(y)
                loss = criterion(y, y_batch[:, step])
                total_loss = total_loss + loss

            total_loss /= model.output_len
            total_loss.backward()
            optimizer.step()

            epoch_loss += total_loss.item()

        scheduler.step()
        if (epoch + 1) % 10 == 0:
            teacher_forcing_ratio *= 0.8

        ret_train_loss = epoch_loss / len(train_data_loader)

        if verbose:
            print(
                f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(train_data_loader):.4f}"
            )

        if (epoch + 1) % 10 == 0 and check_losses:
            with torch.no_grad():
                test_train_loss = 0
                for X_test, y_test in test_data_loader:
                    X_test, y_test = X_test.to(device), y_test.to(device)
                    outputs = model(X_test)
                    if delta:
                        outputs[:, 0] += X_test[:, -1, :]
                        for i in range(1, model.output_len):
                            outputs[:, i] += outputs[:, i - 1]
                    loss = criterion(outputs, y_test)
                    test_train_loss += loss.item()
                test_train_loss /= len(test_data_loader)
                print(f"Current Test loss: {test_train_loss:.4f}")

        if early_stopping and ((epoch + 1) % 10 == 0):
            try:
                val_loader_len = len(test_data_loader)
            except Exception:
                val_loader_len = 0
            if val_loader_len > 0:
                model.eval()
                with torch.no_grad():
                    val_loss = 0.0
                    for X_val, y_val in test_data_loader:
                        X_val, y_val = X_val.to(device), y_val.to(device)
                        outputs = model(X_val)
                        if delta:
                            outputs[:, 0] += X_val[:, -1, :]
                            for i in range(1, model.output_len):
                                outputs[:, i] += outputs[:, i - 1]
                        loss = criterion(outputs, y_val)
                        val_loss += loss.item()
                    val_loss /= val_loader_len
                if epoch == 0:
                    best_val_loss = val_loss
                    epochs_no_improve = 0
                else:
                    if val_loss + early_stopping_min_delta < best_val_loss:
                        best_val_loss = val_loss
                        epochs_no_improve = 0
                    else:
                        epochs_no_improve += 1

                if verbose:
                    print(
                        f"Validation loss: {val_loss:.6f} | Best: {best_val_loss:.6f} | No improve epochs: {epochs_no_improve}"
                    )

                model.train()

                if epochs_no_improve >= early_stopping_patience:
                    if verbose:
                        print(
                            f"Early stopping: no improvement in {early_stopping_patience} epochs. Stopping training."
                        )
                    break

    # --- Final evaluation on test set ---
    model.eval()
    test_loss = 0
    all_targets = []
    all_preds = []

    if verbose:
        for X_test, y_test in test_data_loader:
            X_test, y_test = X_test.to(device), y_test.to(device)
            print(
                f"Input sequence: {X_test[0]}\nTarget sequence: {y_test[0]}\nPredicted sequence: {model(X_test)[0]}"
            )
            break

    with torch.no_grad():
        for X_test, y_test in test_data_loader:
            X_test, y_test = X_test.to(device), y_test.to(device)
            outputs = model(X_test)
            if delta:
                outputs[:, 0] += X_test[:, -1, :]
                for i in range(1, model.output_len):
                    outputs[:, i] += outputs[:, i - 1]
            loss = criterion(outputs, y_test)
            test_loss += loss.item()

            if return_preds:
                # squeeze ultima dimensione se presente: (batch, output_len, 1) -> (batch, output_len)
                preds_np = outputs.squeeze(-1).cpu().numpy()
                targets_np = y_test.squeeze(-1).cpu().numpy()
                all_preds.append(preds_np)
                all_targets.append(targets_np)

    ret_test_loss = test_loss / len(test_data_loader)

    if verbose:
        print(f"Test loss: {test_loss/len(test_data_loader):.4f}")

    if return_preds:
        all_preds = np.concatenate(all_preds, axis=0)  # (N_windows, output_len)
        all_targets = np.concatenate(all_targets, axis=0)  # (N_windows, output_len)
        # restituisce l'ultima finestra = vero orizzonte di forecast
        return ret_train_loss, ret_test_loss, all_targets[-1], all_preds[-1]

    return ret_train_loss, ret_test_loss


def train_encoder_model(
    model: EncoderOnlyModel,
    epochs: int,
    train_data_loader: DataLoader,
    test_data_loader: DataLoader,
    verbose: bool = True,
    teacher_forcing_ratio: float = 1.0,
    pretrain_seca: bool = True,
) -> Tuple[float, float]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    model.train()
    if pretrain_seca:
        model.seca.start()
        model.seca.unfreeze()

    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()

    ret_train_loss = 0.0
    ret_test_loss = 0.0

    for epoch in range(epochs):
        epoch_loss = 0

        for X_batch, y_batch in train_data_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            batch_size, output_len, _ = y_batch.shape

            optimizer.zero_grad()
            total_loss = torch.tensor(0.0, device=device)

            for step in range(model.output_len):
                y = model.single_forward(X_batch)
                X_batch = X_batch[:, 1:]
                if random.random() < teacher_forcing_ratio:
                    X_batch = torch.cat(
                        [X_batch, model.seca.encode(y_batch[:, step].unsqueeze(1))],
                        dim=1,
                    )
                else:
                    X_batch = torch.cat([X_batch, y.unsqueeze(1)], dim=1)
                y = model.seca.decode(y)
                loss = criterion(y, y_batch[:, step])
                loss = torch.sqrt(loss)
                total_loss = total_loss + loss

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += total_loss.item()

        if (epoch + 1) % 5 == 0:
            teacher_forcing_ratio *= 0.9

        ret_train_loss = epoch_loss / len(train_data_loader)

        if verbose:
            print(
                f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(train_data_loader):.4f}"
            )

    model.eval()
    test_loss = 0

    if verbose:
        for X_test, y_test in test_data_loader:
            X_test, y_test = X_test.to(device), y_test.to(device)
            predicted = model(X_test)
            print(
                f"Input sequence: {X_test[0]}\nTarget sequence: {y_test[0]}\nPredicted sequence: {predicted[0]}"
            )
            break

    for X_test, y_test in test_data_loader:
        X_test, y_test = X_test.to(device), y_test.to(device)
        outputs = model(X_test)
        loss = criterion(outputs, y_test)
        loss = torch.sqrt(loss)
        test_loss += loss.item()

    ret_test_loss = test_loss / len(test_data_loader)

    if verbose:
        print(f"Test loss: {test_loss/len(test_data_loader):.4f}")

    return ret_train_loss, ret_test_loss
