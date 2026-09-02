# Predictive Maintenance Benchmark on NASA C-MAPSS

A comparative empirical study of classical machine learning and deep learning models for remaining useful life prediction on the NASA C-MAPSS turbofan engine degradation dataset, with cross-dataset generalization testing on AI4I 2020.

**Course:** CPSC 393 Machine Learning, Spring 2026
**Institution:** Chapman University
**Final Project**

## Project Overview

This project benchmarks four model architectures on a shared evaluation framework: classical tabular methods (Ridge, Logistic Regression, Random Forest, XGBoost), a stacked LSTM, a dilated 1D CNN, and a Transformer encoder. All models are evaluated on identical engine-level splits with shared preprocessing.

**Headline result:** An unweighted ensemble of XGBoost and LSTM regression predictions achieves test RMSE 12.95 on FD001, exceeding the strongest single model by 4.1 percent.

## Setup Instructions

### Dependencies

- Python 3.10 or higher
- PyTorch 2.0+ (with MPS acceleration for Apple Silicon, or CUDA for NVIDIA GPUs)
- XGBoost 2.0+
- scikit-learn 1.3+
- Optuna 3.5+
- SHAP 0.44+
- NumPy, Pandas, Matplotlib, Seaborn

### Installation

```bash
# Clone the repository
git clone https://github.com/melaskary72/predictive-maintenance-cmapss.git
cd predictive-maintenance-cmapss

# Create a virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Dataset Information

### NASA C-MAPSS (Primary Dataset)

The NASA Commercial Modular Aero-Propulsion System Simulation Turbofan Engine Degradation Simulation Dataset is the standard benchmark for predictive maintenance research.

**Download:** https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/

The dataset contains four subsets (FD001 through FD004) varying in operating conditions and fault modes. Place the raw text files in `data/raw/CMAPSSData/` following the structure expected by `src/data/load_cmapss.py`.

### AI4I 2020 (Secondary Dataset)

Used for cross-dataset generalization testing.

**Download:** https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset

Place the CSV in `data/raw/ai4i2020.csv`.

## How to Run the Code

The project is organized into five sequential phases. Each phase produces artifacts consumed by the next.

```bash
# Phase 1: Data loading and exploratory analysis
python -m src.phase1.run_eda

# Phase 2: Classical baselines (Ridge, Logistic Regression, Random Forest, XGBoost)
python -m src.phase2.run_baselines

# Phase 3: Deep models (LSTM, CNN, Transformer) with Optuna optimization
python -m src.phase3.run_deep_models

# Phase 4: Ablation study and ensemble evaluation
python -m src.phase4.run_ablation_and_ensembles

# Phase 5: Horizon sensitivity sweep and AI4I cross-dataset generalization
python -m src.phase5.run_horizon_sweep
python -m src.phase5.run_ai4i_generalization
```

Each phase writes its outputs to the corresponding `results/` subdirectory. The complete pipeline takes approximately 90 minutes on an Apple M-series chip with MPS acceleration.

## Results

### Headline Findings (FD001 Test Set)

| Model | Test RMSE | ROC AUC | PR AUC | F1 | Brier |
|---|---|---|---|---|---|
| Ridge | 17.20 | — | — | — | — |
| Logistic Regression | — | 0.9960 | 0.8918 | 0.8042 | 0.0110 |
| Random Forest | 14.83 | 0.9976 | **0.9326** | 0.8265 | **0.0077** |
| XGBoost | 13.50 | 0.9975 | 0.9290 | 0.8276 | 0.0079 |
| LSTM | 13.59 | 0.9971 | 0.9215 | 0.7310 | 0.0096 |
| CNN | 14.54 | 0.9968 | 0.9050 | 0.8025 | 0.0108 |
| Transformer | 15.24 | 0.9962 | 0.8902 | 0.7780 | 0.0123 |
| **XGB + LSTM ensemble** | **12.95** | 0.9976 | 0.9036 | 0.8438 | 0.0083 |
| XGB + LSTM + CNN ensemble | 13.14 | **0.9977** | 0.9337 | **0.8438** | 0.0083 |

### Principal Findings

1. **Ensemble averaging wins.** XGBoost + LSTM achieves RMSE 12.95, a 4.1% improvement over the best single model.
2. **Hand-engineered features hurt deep models.** Concatenating 84 engineered features to the LSTM raised RMSE from 13.59 to 14.82 (+1.23). CNN saw +0.05.
3. **Transformer underperforms at this scale.** With 30-cycle windows on 80 training engines, the Transformer trails both LSTM and CNN.
4. **Methodology transfers.** Applied to AI4I 2020: test ROC AUC 0.978, 5-fold CV 0.975 ± 0.004.

See the final report PDF for full analysis, interpretability via SHAP, calibration analysis, and discussion of limitations.

## Project Structure

```
predictive-maintenance-cmapss/
├── data/
│   └── raw/                      # Raw datasets (gitignored)
├── src/
│   ├── data/                     # Data loading and preprocessing
│   ├── phase1/                   # EDA
│   ├── phase2/                   # Classical baselines
│   ├── phase3/                   # Deep models
│   ├── phase4/                   # Ablation and ensembles
│   └── phase5/                   # Horizon sweep and AI4I
├── results/
│   ├── eda/                      # Phase 1 outputs
│   ├── figures/                  # Generated plots
│   ├── tables/                   # CSV result tables
│   └── models/                   # Saved model checkpoints
├── requirements.txt
└── README.md
```

## Contributors

**Mohamed El Askary** — Chapman University

This project was completed as a solo submission. I was responsible for the full project lifecycle: dataset selection, exploratory data analysis, pipeline implementation, model development across all four architecture families, hyperparameter tuning, evaluation, generalization testing, and final report preparation.

AI-assisted tooling (Claude) was used for code drafting and prose editing under the author's direction, with all design decisions, model selection, and analytical interpretations made by the author.

## License

This project is submitted as coursework for CPSC 393 at Chapman University. The NASA C-MAPSS dataset and AI4I 2020 dataset have their own licenses; refer to the respective sources.
