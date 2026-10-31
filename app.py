"""Streamlit app for evaluating saved ML assignment classifiers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


PROJECT_ROOT = Path(__file__).parent
MODEL_DIR = PROJECT_ROOT / "model"
REPORTS_DIR = PROJECT_ROOT / "reports"
DEFAULT_TEST_DATA_PATH = PROJECT_ROOT / "test_data.csv"
REGISTRY_PATH = MODEL_DIR / "model_registry.json"


@st.cache_data
def load_registry() -> dict[str, Any]:
    with open(REGISTRY_PATH) as file:
        return json.load(file)


@st.cache_data
def load_default_test_data() -> pd.DataFrame:
    return pd.read_csv(DEFAULT_TEST_DATA_PATH)


@st.cache_data
def load_training_metrics() -> pd.DataFrame | None:
    metrics_path = REPORTS_DIR / "model_metrics.csv"
    if not metrics_path.exists():
        return None
    return pd.read_csv(metrics_path)


@st.cache_resource
def load_preprocessor(preprocessor_file: str):
    return joblib.load(MODEL_DIR / preprocessor_file)


@st.cache_resource
def load_model(model_file: str):
    return joblib.load(MODEL_DIR / model_file)


def validate_input_data(
    data: pd.DataFrame, feature_columns: list[str], target_column: str
) -> list[str]:
    required_columns = feature_columns + [target_column]
    return [column for column in required_columns if column not in data.columns]


def get_positive_class_score(model, features) -> Any:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(features)[:, 1]
    if hasattr(model, "decision_function"):
        return model.decision_function(features)
    return None


def evaluate_model(
    model, transformed_features, y_true: pd.Series
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Any]:
    y_pred = model.predict(transformed_features)
    y_score = get_positive_class_score(model, transformed_features)

    auc = roc_auc_score(y_true, y_score) if y_score is not None else None
    metrics = pd.DataFrame(
        [
            {
                "Accuracy": accuracy_score(y_true, y_pred),
                "AUC": auc,
                "Precision": precision_score(y_true, y_pred, zero_division=0),
                "Recall": recall_score(y_true, y_pred, zero_division=0),
                "F1": f1_score(y_true, y_pred, zero_division=0),
                "MCC": matthews_corrcoef(y_true, y_pred),
            }
        ]
    )

    matrix = pd.DataFrame(
        confusion_matrix(y_true, y_pred),
        index=["Actual 0", "Actual 1"],
        columns=["Predicted 0", "Predicted 1"],
    )
    report = pd.DataFrame(
        classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    ).transpose()

    return metrics, matrix, report, y_pred


def format_metrics_table(metrics: pd.DataFrame) -> pd.DataFrame:
    numeric_columns = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]
    formatted = metrics.copy()
    for column in numeric_columns:
        if column in formatted.columns:
            formatted[column] = formatted[column].map(lambda value: f"{value:.4f}")
    return formatted


def build_distribution(y_true: pd.Series, y_pred) -> pd.DataFrame:
    actual_counts = y_true.value_counts().sort_index()
    predicted_counts = pd.Series(y_pred).value_counts().sort_index()
    return pd.DataFrame(
        {
            "Class": ["Not exited", "Exited"],
            "Actual": [int(actual_counts.get(0, 0)), int(actual_counts.get(1, 0))],
            "Predicted": [
                int(predicted_counts.get(0, 0)),
                int(predicted_counts.get(1, 0)),
            ],
        }
    )


def show_metric_cards(metrics: pd.DataFrame) -> None:
    metric_row = metrics.iloc[0]
    first_row = st.columns(3)
    first_row[0].metric(
        "Accuracy",
        f"{metric_row['Accuracy']:.4f}",
        border=True,
        icon=":material/check_circle:",
        height="stretch",
    )
    first_row[1].metric(
        "AUC",
        f"{metric_row['AUC']:.4f}",
        border=True,
        icon=":material/query_stats:",
        height="stretch",
    )
    first_row[2].metric(
        "Precision",
        f"{metric_row['Precision']:.4f}",
        border=True,
        icon=":material/target:",
        height="stretch",
    )

    second_row = st.columns(3)
    second_row[0].metric(
        "Recall",
        f"{metric_row['Recall']:.4f}",
        border=True,
        icon=":material/visibility:",
        height="stretch",
    )
    second_row[1].metric(
        "F1 score",
        f"{metric_row['F1']:.4f}",
        border=True,
        icon=":material/balance:",
        height="stretch",
    )
    second_row[2].metric(
        "MCC",
        f"{metric_row['MCC']:.4f}",
        border=True,
        icon=":material/analytics:",
        height="stretch",
    )


def main() -> None:
    st.set_page_config(
        page_title="ML assignment 2",
        page_icon=":material/analytics:",
        layout="wide",
    )

    registry = load_registry()
    feature_columns = registry["feature_columns"]
    target_column = registry["target_column"]
    model_files = registry["model_files"]

    with st.sidebar:
        st.header("Evaluation controls", icon=":material/tune:")
        uploaded_file = st.file_uploader(
            "Upload test CSV",
            type=["csv"],
            help=f"CSV must include the input features and `{target_column}`.",
        )
        model_name = st.selectbox("Model", list(model_files.keys()))

        st.subheader("Artifacts", icon=":material/folder:")
        st.badge("Preprocessor ready", icon=":material/check:", color="green")
        st.badge("Models ready", icon=":material/check:", color="green")
        st.badge("Metrics ready", icon=":material/check:", color="green")

        with st.expander("Upload schema", icon=":material/table_chart:"):
            schema = pd.DataFrame(
                {
                    "Column": feature_columns + [target_column],
                    "Purpose": ["Feature"] * len(feature_columns) + ["Target"],
                }
            )
            st.dataframe(
                schema,
                hide_index=True,
                alt="Required CSV columns for uploaded evaluation data.",
            )

    input_data = (
        pd.read_csv(uploaded_file)
        if uploaded_file is not None
        else load_default_test_data()
    )
    data_source = "Uploaded CSV" if uploaded_file is not None else "test_data.csv"

    missing_columns = validate_input_data(input_data, feature_columns, target_column)
    if missing_columns:
        st.error(
            "The selected CSV is missing required columns: "
            + ", ".join(f"`{column}`" for column in missing_columns),
            icon=":material/error:",
        )
        st.stop()

    preprocessor = load_preprocessor(registry["preprocessor_file"])
    model = load_model(model_files[model_name])

    X = input_data[feature_columns]
    y_true = input_data[target_column]
    transformed_features = preprocessor.transform(X)
    metrics, matrix, report, predictions = evaluate_model(
        model, transformed_features, y_true
    )

    st.title(
        "Bank customer churn classifier",
        icon=":material/analytics:",
    )
    st.caption("Saved notebook artifacts are used directly. No retraining runs in the app.")

    with st.container(border=True):
        top_left, top_right = st.columns([1.5, 1], vertical_alignment="center")
        with top_left:
            st.header(model_name, icon=":material/model_training:")
            st.caption(f"Evaluating `{data_source}` with `{target_column}` as target.")
        with top_right:
            selected_metrics = metrics.iloc[0]
            st.metric(
                "Current F1 score",
                f"{selected_metrics['F1']:.4f}",
                border=True,
                icon=":material/balance:",
            )

    overview = st.columns(4)
    overview[0].metric(
        "Rows evaluated",
        f"{len(input_data):,}",
        border=True,
        icon=":material/table_rows:",
        height="stretch",
    )
    overview[1].metric(
        "Input features",
        f"{len(feature_columns):,}",
        border=True,
        icon=":material/view_column:",
        height="stretch",
    )
    overview[2].metric(
        "Actual churn rate",
        f"{y_true.mean() * 100:.1f}%",
        border=True,
        icon=":material/group_remove:",
        height="stretch",
    )
    overview[3].metric(
        "Predicted churn rate",
        f"{pd.Series(predictions).mean() * 100:.1f}%",
        border=True,
        icon=":material/online_prediction:",
        height="stretch",
    )

    st.header("Evaluation metrics", icon=":material/bar_chart:")
    show_metric_cards(metrics)

    left, right = st.columns([1.15, 1])
    with left:
        with st.container(border=True, height="stretch"):
            st.subheader("Class distribution", icon=":material/stacked_bar_chart:")
            distribution = build_distribution(y_true, predictions)
            st.bar_chart(
                distribution,
                x="Class",
                y=["Actual", "Predicted"],
                y_label="Customers",
                alt=f"Actual and predicted class distribution for {model_name}.",
            )

    with right:
        with st.container(border=True, height="stretch"):
            st.subheader("Confusion matrix", icon=":material/grid_view:")
            st.dataframe(
                matrix,
                alt=f"Confusion matrix for {model_name}.",
            )

    with st.container(border=True):
        st.subheader("Model comparison", icon=":material/leaderboard:")
        training_metrics = load_training_metrics()
        if training_metrics is None:
            st.info(
                "Run the training notebook to generate `reports/model_metrics.csv`.",
                icon=":material/info:",
            )
        else:
            st.dataframe(
                format_metrics_table(training_metrics),
                hide_index=True,
                alt="Comparison table of saved classifier metrics from the training notebook.",
            )

    report_col, preview_col = st.columns([1, 1.25])
    with report_col:
        with st.container(border=True, height="stretch"):
            st.subheader("Classification report", icon=":material/summarize:")
            st.dataframe(
                report.round(4),
                alt=f"Classification report for {model_name}.",
            )

    with preview_col:
        with st.container(border=True, height="stretch"):
            st.subheader("Prediction preview", icon=":material/preview:")
            preview = input_data[feature_columns + [target_column]].copy()
            preview["Predicted"] = predictions
            st.dataframe(
                preview.head(50),
                hide_index=True,
                alt=f"First 50 predictions generated by {model_name}.",
            )


if __name__ == "__main__":
    main()
