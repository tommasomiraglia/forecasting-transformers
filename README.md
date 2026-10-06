# Forecasting with Transformers

Code for my Bachelor's thesis in Computer Science at the University of Bologna (**Università di Bologna**): an experimental comparison of Transformer-based models and large pretrained models for time series forecasting, evaluated mainly on the M4 and M3 benchmark datasets. 

This repository is a fork of [FGaragnani/deconstructing-transformers](https://github.com/FGaragnani/deconstructing-transformers), from whose codebase my thesis project started. The thesis itself (written in Italian) is available in the `Thesis` repository (see also the [UniBo CRIS institutional repository](https://cris.unibo.it/handle/11585/1028028)).

---

## What I Did
* Trained and evaluated the minimalist Transformer against large pretrained forecasting models (Chronos, TimeGPT) on the M3 and M4 benchmark datasets.
* Performed attention mechanism experiments to investigate seasonality capture and representation learning across the Transformer's early layers.
* Ran ablation and hyperparameter studies (using Optuna) alongside scalability analyses on longer series.
* Applied the models to a real-world dataset analyzing restaurant search trends from Google Trends (`ristorantiGTrend.csv`).

---

## What's in the Repo

* **Model comparison on M4/M3:** `chronos_on_M4.py`, `chronos_on_M3.py`, `timegpt_on_M4.py`, and `comparisons.py` run pretrained forecasting models and compare performance metrics (RMSE / NRMSE, saved as CSV files in the root and in `results/`).
* **Ablation and tuning:** `ablation_study.py` and `optuna_study.py` (automated hyperparameter search with Optuna).
* **Analysis & Visualisation:** `attention_plot.py` and `attention_view.py` (attention weight visualization), plus `scalability.py` and `long_series.py` (evaluating behavior on longer series and larger inputs).
* **Real-world test:** `ristoranti.py` applies a trained model to restaurant-related Google Trends data.
* **Model code:** `vanilla/`, `tcan/`, and `statpred/` hold the custom model implementations and statistical baselines.

---

## Setup

```bash
pip install -r requirements.txt
