import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
)


# ============================================================
# Page configuration
# ============================================================
st.set_page_config(
    page_title="2025AC05048",
    page_icon=":robot:"
)

# ============================================================
# Simple styling
# ============================================================
st.markdown(
    """
    <style>
        .block-container {
            max-width: 1150px;
            padding-top: 2rem;
        }

        .card {
            background-color: #13263a;
            border: 1px solid #263247;
            border-radius: 12px;
            padding: 1rem;
            text-align: center;
        }

        .card-label {
            color: #94a3b8;
            font-size: 0.8rem;
            text-transform: uppercase;
        }

        .card-value {
            color: #f8fafc;
            font-size: 1.6rem;
            font-weight: 700;
            margin-top: 0.2rem;
        }

        .winner {
            background-color: #13263a;
            border: 1px solid #2f6f9f;
            border-radius: 12px;
            padding: 1rem 1.2rem;
            margin: 1rem 0;
        }

        .winner-title {
            color: #7dd3fc;
            font-weight: 700;
        }

        .winner-text {
            color: #e2e8f0;
            margin-top: 0.25rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Models and Results File
MODEL_DIR = "model"
RESULTS_FILE = os.path.join("results", "model_comparison.csv")

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Random Forest": "random_forest.joblib",
}

# Helper functions
@st.cache_data
def load_results():
    if not os.path.exists(RESULTS_FILE):
        return pd.DataFrame()

    df = pd.read_csv(RESULTS_FILE)

    # Make column names tolerant of minor formatting differences.
    rename_map = {}
    for column in df.columns:
        key = column.strip().lower()

        if key == "ml model name":
            rename_map[column] = "ML Model Name"
        elif key == "accuracy":
            rename_map[column] = "Accuracy"
        elif key == "auc":
            rename_map[column] = "AUC"
        elif key == "precision":
            rename_map[column] = "Precision"
        elif key == "recall":
            rename_map[column] = "Recall"
        elif key == "f1":
            rename_map[column] = "F1"
        elif key == "mcc":
            rename_map[column] = "MCC"

    return df.rename(columns=rename_map)

@st.cache_resource
def load_model(model_name):
    path = os.path.join(MODEL_DIR, MODEL_FILES[model_name])

    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")

    return joblib.load(path)

def find_target_column(df):
    possible_names = [
        "income",
        "Income",
        "income ",
        "class",
        "target",
    ]

    for column in possible_names:
        if column in df.columns:
            return column

    return None

def calculate_metrics(y_true, predictions, probabilities):
    y_binary = (y_true == ">50K").astype(int)

    return {
        "Accuracy": accuracy_score(y_true, predictions),
        "AUC": roc_auc_score(y_binary, probabilities),
        "Precision": precision_score(
            y_true,
            predictions,
            pos_label=">50K",
            zero_division=0,
        ),
        "Recall": recall_score(
            y_true,
            predictions,
            pos_label=">50K",
            zero_division=0,
        ),
        "F1": f1_score(
            y_true,
            predictions,
            pos_label=">50K",
            zero_division=0,
        ),
        "MCC": matthews_corrcoef(y_true, predictions),
    }

# Header
st.title("2025AC05048 - Adult Income Classification")

st.markdown(
    """
    **Machine Learning Model Comparison using the UCI Adult Income Dataset**

    This application evaluates five classification algorithms for predicting whether an individual's annual income is **≤50K or >50K**.
    """
)

st.divider()

# Dataset overview
st.subheader("Dataset Overview")

c1, c2, c3, c4 = st.columns(4)

overview = [
    ("Training Samples", "32,561"),
    ("Test Samples", "16,281"),
    ("Features", "14"),
    ("Classes", "2"),
]

for column, (label, value) in zip([c1, c2, c3, c4], overview):
    with column:
        st.markdown(
            f"""
            <div class="card">
                <div class="card-label">{label}</div>
                <div class="card-value">{value}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ============================================================
# Model comparison
# ============================================================
st.subheader("Model Comparison")

results = load_results()

if results.empty:
    st.error("Model comparison results were not found. ")
else:
    required_columns = [
        "ML Model Name",
        "Accuracy",
        "AUC",
        "Precision",
        "Recall",
        "F1",
        "MCC"
    ]

    missing = [column for column in required_columns if column not in results.columns]

    if missing:
        st.error(
            "The model comparison file is missing these columns: "
            + ", ".join(missing)
        )
    else:
        display_results = results[required_columns].copy()

        # Convert metrics to percentages for easier reading.
        for column in required_columns[1:]:
            display_results[column] = (
                display_results[column].astype(float) * 100
            ).round(2)

        display_results = display_results.rename(
            columns={
                "ML Model Name": "Model",
                "Accuracy": "Accuracy (%)",
                "AUC": "AUC (%)",
                "Precision": "Precision (%)",
                "Recall": "Recall (%)",
                "F1": "F1 (%)",
                "MCC": "MCC (%)",
            }
        )

        st.dataframe(display_results, hide_index=True, use_container_width=True)

        # Overall winner based on the same comparison used in training.
        winner = results.loc[results["F1"].astype(float).idxmax()]

        st.markdown(
            f"""
            <div class="winner">
                <div class="winner-title">Overall Best Model</div>
                <div class="winner-text">
                    <strong>{winner["ML Model Name"]}</strong>
                    — Accuracy: {float(winner["Accuracy"]):.2%},
                    AUC: {float(winner["AUC"]):.2%},
                    F1: {float(winner["F1"]):.2%},
                    MCC: {float(winner["MCC"]):.2%}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )