# LogisChain AI

### Predictive Trade Finance & Logistics Valuation

LogisChain AI is a prototype analytics platform combining machine
learning, working-capital calculations, and logistics disruption
simulation to explore financial and operational risk.

## Features

- Credit-risk classification using Random Forest and Gradient Boosting.
- Model comparison using accuracy, ROC-AUC, precision, recall, and F1.
- Interactive credit-risk prediction interface.
- Cash Conversion Cycle (CCC) calculation.
- Net Working Capital and Current Ratio calculations.
- Logistics disruption simulation for delivery delays and freight costs.
- Data validation, preprocessing, and automated tests.
- Streamlit dashboard for interactive exploration.

## Technology Stack

- Python 3.11
- Pandas and NumPy
- Scikit-learn and XGBoost
- Streamlit
- Matplotlib, Seaborn, and Plotly
- Pytest and Joblib

## Dataset

The current credit-risk baseline uses the UCI Statlog German Credit
dataset, containing 1,000 historical records.

Dataset source:
https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data

**Important:** This is a personal credit benchmark, not a trade-finance
dataset. Results do not establish performance on trade-finance applicants.

## Baseline Results

Random Forest:

- Accuracy: 75.5%
- ROC-AUC: 0.7907
- Bad-credit precision: 58.21%
- Bad-credit recall: 65.00%
- Bad-credit F1: 61.42%

Gradient Boosting:

- Accuracy: 76.5%
- ROC-AUC: 0.7815
- Bad-credit precision: 65.85%
- Bad-credit recall: 45.00%
- Bad-credit F1: 53.47%

Evaluation uses a held-out test set of 200 records. Results are
benchmark-specific and should not be interpreted as validated
real-world lending performance.

## Installation

Use Python 3.11.

Create and activate a virtual environment on Windows:

```cmd
py -3.11 -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

Download the German Credit dataset from the UCI source above and
place `german.data` in `data/raw/`.

## Run the Application

From the project root:

```cmd
streamlit run app.py
```

## Train the Baseline Model

```cmd
python -m src.models.train_credit_model
```

## Compare Models

```cmd
python -m src.models.compare_models
```

## Run Tests

```cmd
python -m pytest tests -v
```

## Limitations

- The current classifier uses a historical personal-credit benchmark.
- No validated trade-finance training dataset is currently integrated.
- Logistics simulation outputs are hypothetical scenario estimates,
  not live forecasts.
- Financial metrics are arithmetic indicators, not financial advice.
- Model predictions must not be used as the sole basis for lending
  or other high-impact decisions.

## Project Status

Interactive prototype with data-processing modules, baseline model
evaluation, financial calculators, logistics simulation, and automated
tests.
