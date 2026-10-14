# Bank Marketing Subscription Prediction

## Project Overview
This project develops a machine-learning pipeline to predict whether a bank customer will subscribe to a term deposit.

## Dataset
- Source: UCI Machine Learning Repository — Bank Marketing
- Original dataset: 45,211 records
- Training sample: approximately 12,000 records
- Target: subscription (`yes` or `no`)

## Methodology
1. Data cleaning and feature engineering.
2. Exclusion of call duration to reduce target leakage.
3. Stratified train/test split.
4. Numeric and categorical preprocessing.
5. Comparison of Logistic Regression, Decision Tree and Random Forest.
6. GridSearchCV with stratified 3-fold cross-validation.
7. Evaluation using accuracy, precision, recall, F1 score and ROC-AUC.
8. Flask REST API for predictions.

## Results

| Model | Accuracy | F1 Score | ROC-AUC |
|---|---:|---:|---:|
| Logistic Regression | 76.71% | 0.3768 | 0.7619 |
| Decision Tree | 74.21% | 0.3408 | 0.6981 |
| Random Forest | 87.08% | 0.4542 | 0.7943 |

Random Forest achieved the highest cross-validation F1 score of 0.4223 and was selected as the final model.

## API Endpoints
- `GET /` — project information
- `GET /health` — health check
- `POST /predict` — predict subscription from customer data

The API was tested locally on port 5001.

## Limitations
The target classes are imbalanced. Although Random Forest achieved the highest accuracy, its F1 score shows that positive-subscription prediction remains challenging. Further tuning and threshold evaluation could improve performance.

## Run Locally
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python train.py
python app.py
```

## Dataset Source
UCI Machine Learning Repository: https://archive.ics.uci.edu/dataset/222/bank+marketing