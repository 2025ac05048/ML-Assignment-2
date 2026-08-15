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
    classification_report
)


# Page configuration
st.set_page_config(
    page_title="2025AC05048",
    page_icon=":robot:"
)

# Styling
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
    unsafe_allow_html=True
)

# Models and Results File
MODEL_DIR = "model"
RESULTS_FILE = os.path.join("results", "model_comparison.csv")

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Random Forest": "random_forest.joblib"
}

# CONSTANTS
COLUMNS = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education-num",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
    "native-country",
    "income"
]

FEATURE_COLUMNS = COLUMNS[:-1]

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

def clean_uploaded_data(data):
    data = data.copy()
    data.columns = [str(col).strip() for col in data.columns]

    if "income" in data.columns:
        data["income"] = (data["income"].astype(str).str.strip().str.replace(".", "", regex=False))

    for col in data.select_dtypes(include=["object"]).columns:
        data[col] = data[col].astype(str).str.strip()

    return data

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
        "MCC": matthews_corrcoef(y_true, predictions)
    }

def evaluate_model(model_name, X, y_true):
    model = load_model(model_name)

    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]

    metrics = calculate_metrics(y_true, predictions, probabilities)

    return model, predictions, probabilities, metrics

# Header
st.title(":robot: Adult Income Classification")

st.markdown(
    """
    **Machine Learning Model Comparison using the <a href="https://archive.ics.uci.edu/ml/datasets/adult" target="_blank">UCI Adult Income Dataset</a>**.

    This application evaluates five classification algorithms for predicting whether an individual's annual income is **≤50K or >50K**.
    The .joblib files under the <a href="https://github.com/2025ac05048/ML-Assignment-2/tree/b5e05f9381607f924e0b955df570f01974fc35d2/model" target="_blank">model</a> directory contain the complete preprocessing and model pipeline.
    """,
    unsafe_allow_html=True,
)

st.divider()

# Dataset overview
st.subheader("Training Dataset Overview")

c1, c2, c3 = st.columns(3)

overview = [
    ("Training Samples", "32,561"),
    ("Features", "14"),
    ("Classes", "2")
]

for column, (label, value) in zip([c1, c2, c3], overview):
    with column:
        st.markdown(
            f"""
            <div class="card">
                <div class="card-label">{label}</div>
                <div class="card-value">{value}</div>
            </div>
            """,
            unsafe_allow_html=True
        )


# Model comparison
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

        for column in required_columns[1:]:
            display_results[column] = (display_results[column].astype(float) * 100).round(2)

        display_results = display_results.rename(
            columns={
                "ML Model Name": "Model",
                "Accuracy": "Accuracy (%)",
                "AUC": "AUC (%)",
                "Precision": "Precision (%)",
                "Recall": "Recall (%)",
                "F1": "F1 (%)",
                "MCC": "MCC (%)"
            }
        )

        st.dataframe(display_results, hide_index=True, use_container_width=True)

        # Overall best model
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
            unsafe_allow_html=True
        )


# Prediction
st.subheader("Prediction")

selected_model = st.selectbox("Select Model", list(MODEL_FILES.keys()))
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is None:
    st.warning("Please upload a CSV file to begin evaluation.")
    st.stop()

try:
    data = pd.read_csv(uploaded_file)
    data = clean_uploaded_data(data)
except Exception as exc:
    st.error(f"Unable to read the CSV file: {exc}")
    st.stop()

missing_features = [
    col for col in FEATURE_COLUMNS
    if col not in data.columns
]

if missing_features:
    st.error(
        "The uploaded file is missing the following required columns: "
        + ", ".join(missing_features)
    )
    st.stop()

X = data[FEATURE_COLUMNS].copy()
for col in X.select_dtypes(include=["object"]).columns:
    X[col] = X[col].replace("?", pd.NA)


# TEST DATASET SUMMARY
st.subheader("1. Test Dataset Summary")

m1, m2, m3 = st.columns(3)

moverview = [
    ("Records", f"{len(data):,}"),
    ("Features", len(FEATURE_COLUMNS)),
    ("Target Column", "income" if "income" in data.columns else "Not provided")
]

for mcolumn, (mlabel, mvalue) in zip([m1, m2, m3], moverview):
    with mcolumn:
        st.markdown(
            f"""
            <div class="card">
                <div class="card-label">{mlabel}</div>
                <div class="card-value">{mvalue}</div>
            </div>
            """,
            unsafe_allow_html=True
        )


# TARGET VALIDATION
has_target = "income" in data.columns

if not has_target:
    st.warning(
        "No `income` target column was found. Predictions can be generated, "
        "but evaluation metrics cannot be calculated."
    )

# SELECTED MODEL EVALUATION
st.subheader(f"2. Results — {selected_model}")

try:
    selected_model_object = load_model(selected_model)
    selected_predictions = selected_model_object.predict(X)
    selected_probabilities = selected_model_object.predict_proba(X)[:, 1]

except Exception as exc:
    st.error("Prediction failed.")
    st.exception(exc)
    st.stop()

# PREDICTION RESULTS
result_data = data.copy()
result_data["Predicted Income"] = selected_predictions

display_columns = FEATURE_COLUMNS + ["Predicted Income"]

if has_target:
    display_columns.append("income")

st.dataframe(
    result_data[display_columns].head(100),
    use_container_width=True,
    height=400
)

st.caption(
    "Showing the first 100 prediction rows. The complete dataset is used "
    "for evaluation."
)

# METRICS
if has_target:
    valid_target = data["income"].isin(["<=50K", ">50K"])

    if not valid_target.all():
        st.warning(
            f"{(~valid_target).sum()} row(s) contain invalid target labels "
            "and will be excluded from evaluation."
        )

    y_true = data.loc[valid_target, "income"]
    predictions = pd.Series(selected_predictions, index=data.index).loc[valid_target]

    probabilities = pd.Series(selected_probabilities, index=data.index).loc[valid_target].to_numpy()

    if len(y_true) == 0:
        st.error("No valid target labels are available for evaluation.")
        st.stop()

    metrics = calculate_metrics(y_true, predictions, probabilities)

    c1, c2, c3 = st.columns(3)
    c4, c5, c6 = st.columns(3)

    c1.metric("Accuracy", f"{metrics['Accuracy']:.4f}")
    c2.metric("AUC", f"{metrics['AUC']:.4f}")
    c3.metric("Precision", f"{metrics['Precision']:.4f}")
    c4.metric("Recall", f"{metrics['Recall']:.4f}")
    c5.metric("F1 Score", f"{metrics['F1']:.4f}")
    c6.metric("MCC", f"{metrics['MCC']:.4f}")


    # CONFUSION MATRIX
    st.subheader("3. Confusion Matrix")

    labels = ["<=50K", ">50K"]

    cm = confusion_matrix(y_true, predictions, labels=labels)

    fig, ax = plt.subplots(figsize=(5, 5))
    image = ax.imshow(cm)

    ax.set_xticks(range(2))
    ax.set_yticks(range(2))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted Income")
    ax.set_ylabel("Actual Income")
    ax.set_title(f"Confusion Matrix — {selected_model}")

    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center")

    fig.colorbar(image, ax=ax)
    st.pyplot(fig)
    plt.close(fig)

    # CLASSIFICATION REPORT
    st.subheader("4. Classification Report")

    report = classification_report(
        y_true,
        predictions,
        labels=labels,
        target_names=labels,
        output_dict=True,
        zero_division=0
    )

    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df.round(4), use_container_width=True)

    # ALL-MODEL COMPARISON
    st.subheader("5. Model Comparison on Uploaded Test Data")

    comparison_rows = []
    with st.spinner("Evaluating all five models..."):
        for model_name in MODEL_FILES:
            try:
                _, pred, prob, model_metrics = evaluate_model(
                    model_name,
                    X.loc[valid_target],
                    y_true
                )

                comparison_rows.append(
                    {
                        "ML Model Name": model_name,
                        **model_metrics
                    }
                )

            except Exception as exc:
                st.warning(f"Could not evaluate {model_name}: {exc}")
                st.exception(exc)

    comparison_df = pd.DataFrame(comparison_rows)

    if not comparison_df.empty:
        comparison_df = comparison_df.sort_values(by="F1", ascending=False)

        st.dataframe(
            comparison_df.style.format(
                {
                    "Accuracy": "{:.4f}",
                    "AUC": "{:.4f}",
                    "Precision": "{:.4f}",
                    "Recall": "{:.4f}",
                    "F1": "{:.4f}",
                    "MCC": "{:.4f}"
                }
            ),
            use_container_width=True
        )

        # F1 comparison chart
        st.subheader("6. F1 Score Comparison")

        chart_df = comparison_df.set_index("ML Model Name")["F1"]

        fig2, ax2 = plt.subplots(figsize=(9, 4))
        chart_df.plot(kind="bar", ax=ax2)

        ax2.set_xlabel("All Machine Learning Models")
        ax2.set_ylabel("F1 Score")
        ax2.set_title("F1 Score Comparison")
        ax2.tick_params(axis="x", rotation=30)

        fig2.tight_layout()
        st.pyplot(fig2)
        plt.close(fig2)


# Footer
st.markdown("---")
st.caption(
    "Adult Income Classification | Logistic Regression | Decision Tree | "
    "KNN | Naive Bayes | Random Forest"
)

st.caption("Name: SHIVAM GUHA | Roll No: 2025AC05048 ")
