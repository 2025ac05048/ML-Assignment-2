# 2025AC05048
import os
import warnings
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix, classification_report

warnings.filterwarnings("ignore")
TRAIN_FILE = "adult.data"
TEST_FILE = "adult.test"
MODEL_DIR = "model"
RESULT_DIR = "results"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

COLUMNS = ["age","workclass","fnlwgt","education","education-num","marital-status","occupation","relationship","race","sex","capital-gain","capital-loss","hours-per-week","native-country","income"]

# Load Training and Test Data
print("=" * 70)
print("Loading Adult Income dataset")
print("=" * 70)
train_df = pd.read_csv(TRAIN_FILE, names=COLUMNS, skipinitialspace=True, na_values="?")
print(f"Training shape: {train_df.shape}")
test_df = pd.read_csv(TEST_FILE, names=COLUMNS, skipinitialspace=True, skiprows=1, na_values="?")
print(f"Test shape: {test_df.shape}")

# Pre-Processing Data
train_df["income"] = train_df["income"].str.strip()
test_df["income"] = test_df["income"].str.strip().str.replace(".", "", regex=False)
valid_labels = ["<=50K", ">50K"]
train_df = train_df[train_df["income"].isin(valid_labels)].copy()
test_df = test_df[test_df["income"].isin(valid_labels)].copy()

X_train = train_df.drop(columns=["income"])
y_train = train_df["income"]
X_test = test_df.drop(columns=["income"])
y_test = test_df["income"]

numeric_features = ["age","fnlwgt","education-num","capital-gain","capital-loss","hours-per-week"]
categorical_features = ["workclass","education","marital-status","occupation","relationship","race","sex","native-country"]

numeric_transformer = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
categorical_transformer = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
preprocessor = ColumnTransformer([("numeric", numeric_transformer, numeric_features), ("categorical", categorical_transformer, categorical_features)])

# Defining Models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Naive Bayes": GaussianNB(),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
}

results = []
print("\n" + "=" * 70)
print("TRAINING AND EVALUATION")
print("=" * 70)

for model_name, classifier in models.items():
    print("\n" + "-" * 70)
    print(f"Training: {model_name}")
    print("-" * 70)
    pipeline = Pipeline([("preprocessor", preprocessor), ("classifier", classifier)])
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_probability = pipeline.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score((y_test == ">50K").astype(int), y_probability)
    precision = precision_score(y_test, y_pred, pos_label=">50K", zero_division=0)
    recall = recall_score(y_test, y_pred, pos_label=">50K", zero_division=0)
    f1 = f1_score(y_test, y_pred, pos_label=">50K", zero_division=0)
    mcc = matthews_corrcoef(y_test, y_pred)
    results.append({"ML Model Name": model_name, "Accuracy": accuracy, "AUC": auc, "Precision": precision, "Recall": recall, "F1": f1, "MCC": mcc})
    
    filename = model_name.lower().replace(" ", "_").replace("-", "_")
    model_path = os.path.join(MODEL_DIR, f"{filename}.joblib")
    joblib.dump(pipeline, model_path)

    print(f"Saved: {model_path}")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"AUC      : {auc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1       : {f1:.4f}")
    print(f"MCC      : {mcc:.4f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred, labels=["<=50K", ">50K"]))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

results_df = pd.DataFrame(results).sort_values(by="F1", ascending=False)
results_path = os.path.join(RESULT_DIR, "model_comparison.csv")
results_df.to_csv(results_path, index=False)
joblib.dump(preprocessor, os.path.join(MODEL_DIR, "preprocessor.joblib"))

print("\n" + "=" * 70)
print("FINAL MODEL COMPARISON")
print("=" * 70)
print(results_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
print("\n" + "=" * 70)
print("TRAINING COMPLETE")
print("=" * 70)
print(f"Models saved in : {MODEL_DIR}")
print(f"Results saved in: {results_path}")
