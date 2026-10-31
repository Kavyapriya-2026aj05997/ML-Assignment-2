# ML Assignment 2 - Classification Model Deployment

## a. Problem statement

This project predicts whether a bank customer is likely to exit or churn based on customer profile, account, and activity details. The task is a binary classification problem where the target column is `Exited`:

- `0`: Customer did not exit
- `1`: Customer exited

The trained models are used in a Streamlit app to evaluate uploaded test data without retraining.

## b. Dataset description

Dataset used: Bank customer churn dataset (`Churn_Modelling.csv`)

- Total instances: 10,000
- Total columns in original dataset: 14
- Target column: `Exited`
- Class distribution:
  - Not exited (`0`): 7,963
  - Exited (`1`): 2,037
- Categorical columns used: `Geography`, `Gender`
- Identifier columns removed before training: `RowNumber`, `CustomerId`, `Surname`
- Input features used for modeling: 10 original feature columns, expanded to 13 processed features after one-hot encoding

The dataset satisfies the assignment requirement of at least 500 instances. The original dataset has 13 non-target columns before removing identifier columns, and the processed training matrix has 13 model-ready features after encoding.

## c. GitHub repository link

https://github.com/Kavyapriya-2026aj05997/ML-Assignment-2

## Live Streamlit app link

https://kavya-ml-assignment-2.streamlit.app/

## d. Models used

All five required classification models were trained and evaluated on the same train/test split:

- Logistic Regression
- Decision Tree Classifier
- K-Nearest Neighbor Classifier
- Naive Bayes Classifier
- Support Vector Machine

## Model comparison table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.7135 | 0.7772 | 0.3872 | 0.7002 | 0.4987 | 0.3482 |
| Decision Tree | 0.7770 | 0.6652 | 0.4543 | 0.4767 | 0.4652 | 0.3246 |
| KNN | 0.8350 | 0.7730 | 0.6638 | 0.3833 | 0.4860 | 0.4172 |
| Naive Bayes | 0.8060 | 0.7773 | 0.5331 | 0.3759 | 0.4409 | 0.3351 |
| Support Vector Machine | 0.8525 | 0.8545 | 0.7044 | 0.4742 | 0.5668 | 0.4957 |

## Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Achieved high recall, which means it identified many churned customers, but precision and accuracy were lower compared with the best model. |
| Decision Tree | Produced moderate results but had lower AUC and MCC, suggesting weaker generalization on the test data. |
| KNN | Achieved good accuracy and precision, but recall was low, meaning it missed more churned customers. |
| Naive Bayes | Performed reasonably on AUC but had the lowest F1 score among the models, mainly due to lower recall. |
| Support Vector Machine | Delivered the best overall balance with the highest accuracy, AUC, F1 score, and MCC. |
| Overall winner for this dataset | Support Vector Machine |

## Repository structure

```text
project-folder/
|-- app.py
|-- requirements.txt
|-- README.md
|-- test_data.csv
|-- sample_upload_test_data.csv
|-- data/
|   |-- Churn_Modelling.csv
|   |-- raw/
|   |-- processed/
|-- model/
|-- notebooks/
|   |-- model_training.ipynb
|-- reports/
|-- .streamlit/
|   |-- config.toml
```

## Streamlit app features

The deployed app includes:

- CSV upload option for test data
- Model selection dropdown
- Saved model loading from `model/`
- Saved preprocessing artifact loading from `model/preprocessor.joblib`
- Evaluation without retraining
- Accuracy, AUC, Precision, Recall, F1, and MCC display
- Confusion matrix
- Classification report
- Prediction preview
- Model comparison table

## How to run locally

```bash
source .venv/bin/activate
streamlit run app.py
```
