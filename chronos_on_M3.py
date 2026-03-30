# Utilizzato per fare qualche test veloce su serie M3 utilizzando chronos stessa struttura di comparisons.py 

import torch
import pandas as pd
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
from chronos import ChronosPipeline
from tqdm import tqdm


class ChronosEvaluator:
    """
    Carica serie temporali da un file Excel, esegue la valutazione con Chronos,
    calcola le metriche e salva i grafici.
    """

    SHEET_MONTHLY = 2

    def __init__(
        self,
        xls_path: str,
        indices: list,
        output_len: int = 18,
        sheet_idx: int = SHEET_MONTHLY,
        model_name: str = "amazon/chronos-t5-small",
        results_dir: str = "results",
    ):
        self.xls_path = xls_path
        self.indices = indices
        self.output_len = output_len
        self.sheet_idx = sheet_idx
        self.results_dir = results_dir

        print(f"Caricamento modello {model_name}...")
        self.model = ChronosPipeline.from_pretrained(
            model_name,
            device_map="cpu",
            torch_dtype=torch.bfloat16,
        )

        self._series: dict = {}  # {dataset_id: np.ndarray}
        self._results: list = []  # lista di dict con le metriche

    # CARICAMENTO DATI

    def load_data(self) -> "ChronosEvaluator":
        """Legge l'Excel e carica le serie corrispondenti agli indici richiesti."""
        print(f"Caricamento dati da {self.xls_path}...")
        df = pd.read_excel(self.xls_path, sheet_name=self.sheet_idx, header=None)
        data_rows = df.iloc[1:]

        for _, row in data_rows.iterrows():
            raw_id = str(row.iloc[0])[1:]
            try:
                idx = int(raw_id)
            except ValueError:
                continue

            if idx not in self.indices:
                continue

            values = row.iloc[6:].to_numpy(dtype=np.float32)
            values = values[~np.isnan(values)]
            self._series[idx] = values

        print(f"  Serie caricate: {list(self._series.keys())}")
        return self

    # METRICHE

    @staticmethod
    def _rmse(actual, forecast):
        actual, forecast = np.array(actual), np.array(forecast)
        return np.sqrt(np.mean((forecast - actual) ** 2))

    @staticmethod
    def _nrmse(actual, forecast):
        rmse = ChronosEvaluator._rmse(actual, forecast)
        r = np.max(actual) - np.min(actual)
        return rmse / r if r != 0 else 0.0

    # PLOT

    def _plot(self, context, actual, forecast, dataset_id):
        n_ctx = len(context)
        n_fct = len(actual)

        x_ctx = np.arange(n_ctx)
        x_fct = np.arange(n_ctx, n_ctx + n_fct)

        plt.figure(figsize=(12, 4))
        plt.plot(x_ctx, context, color="steelblue", label="Context (input)")
        plt.plot(x_fct, actual, color="green", label="Actual")
        plt.plot(x_fct, forecast, color="red", linestyle="--", label="Chronos Forecast")
        plt.axvline(x=n_ctx - 0.5, color="gray", linestyle=":", linewidth=1)
        plt.title(f"Dataset {dataset_id}")
        plt.legend()
        plt.tight_layout()

        path = os.path.join(self.results_dir, f"forecast_{dataset_id}.png")
        plt.savefig(path, dpi=100)
        plt.close()
        print(f"  → Salvato {path}")

    # VALUTAZIONE

    def evaluate(self, plot: bool = True) -> "ChronosEvaluator":
        """Esegue il forecast su tutte le serie caricate e calcola le metriche."""
        if not self._series:
            raise RuntimeError("Nessuna serie caricata. Chiama load_data() prima.")

        self._results = []

        for dataset_id in tqdm(self.indices, desc="Evaluating Chronos"):
            if dataset_id not in self._series:
                print(f"  Serie {dataset_id} non trovata, skip.")
                continue

            series = self._series[dataset_id]

            if len(series) <= self.output_len:
                print(f"  Serie {dataset_id} troppo corta ({len(series)}), skip.")
                continue

            context_np = series[: -self.output_len]
            target_np = series[-self.output_len :]
            context = torch.tensor(context_np, dtype=torch.float32)

            with torch.no_grad():
                forecast_samples = self.model.predict(
                    context.unsqueeze(0), self.output_len
                )
                forecast_np = forecast_samples.mean(dim=1).squeeze(0).cpu().numpy()

            if plot:
                self._plot(context_np, target_np, forecast_np, dataset_id)

            self._results.append(
                {
                    "dataset_id": dataset_id,
                    "RMSE": self._rmse(target_np, forecast_np),
                    "NRMSE": self._nrmse(target_np, forecast_np),
                    "context_len": len(context_np),
                }
            )

        return self

    # RISULTATI

    def results(self) -> pd.DataFrame:
        """Restituisce i risultati come DataFrame."""
        return pd.DataFrame(self._results)

    def save_results(self, filename: str = "chronos_metrics.csv") -> "ChronosEvaluator":
        """Salva i risultati in un file CSV."""
        path = os.path.join(self.results_dir, filename)
        self.results().to_csv(path, index=False)
        print(f"Risultati salvati in {path}")
        return self

    def print_summary(self) -> "ChronosEvaluator":
        """Stampa un sommario delle metriche."""
        df = self.results()
        print("\nChronos Evaluation Results:")
        print("=" * 50)
        print(df.to_string(index=False))
        print(f"\nMedia RMSE  : {df['RMSE'].mean():.4f}")
        print(f"Media NRMSE : {df['NRMSE'].mean():.4f}")
        return self


# MAIN

if __name__ == "__main__":
    torch.manual_seed(42)

    INDICES = [1652, 1546, 1894, 2047, 2255, 2492, 2594, 2658, 2737, 2758, 2817, 2823]

    (
        ChronosEvaluator(
            xls_path="M3C.xls",
            indices=INDICES,
            output_len=18,
        )
        .load_data()
        .evaluate(plot=False)
        .save_results()
        .print_summary()
    )
