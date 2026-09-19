# 🔧 Motor Failure Prediction API

Predict electric motor failures from sensor data using XGBoost.

## Problem
Unexpected motor failures cost manufacturing plants millions in downtime.
This model predicts failures before they happen.

## Data
- UCI AI4I 2020 Predictive Maintenance Dataset
- 10,000 records, 5 sensor features

## Model Performance
| Metric | Score |
|--------|-------|
| ROC-AUC | 0.965 |
| Recall | 79% |
| Best Threshold | 0.70 |

## Tech Stack
- Python, Pandas, NumPy
- XGBoost, Scikit-learn
- FastAPI, Pydantic
- Docker

## API Endpoints
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check |
| `/predict` | POST | Predict failure from sensor data |

## Run Locally
```bash
pip install -r requirements.txt
python -m uvicorn main:app --reload
