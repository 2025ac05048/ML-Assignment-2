# Name: SHIVAM GUHA | Roll No.: 2025AC05048
# Adult Income Classification

## Overview

This assignment builds and evaluates machine learning classification models using the UCI Adult Income dataset.

The objective is to predict whether an individual's annual income is:

- `<=50K`
- `>50K`

The assignment includes preprocessing, model training, evaluation, and an interactive Streamlit application.

## Dataset description

| Variable         | Type            | Description / Possible Values                                                                                                                                                                                            | Missing Values |
| ---------------- | --------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | -------------- |
| `age`            | Integer         | Age of the individual                                                                                                                                                                                                    | No             |
| `workclass`      | Categorical     | Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked                                                                                                                    | Yes            |
| `fnlwgt`         | Integer         | Final sampling weight                                                                                                                                                                                                    | No             |
| `education`      | Categorical     | Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool                                                                    | No             |
| `education-num`  | Integer         | Numerical representation of education level                                                                                                                                                                              | No             |
| `marital-status` | Categorical     | Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse                                                                                                                | No             |
| `occupation`     | Categorical     | Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces | Yes            |
| `relationship`   | Categorical     | Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried                                                                                                                                                       | No             |
| `race`           | Categorical     | White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black                                                                                                                                                              | No             |
| `sex`            | Binary          | Female, Male                                                                                                                                                                                                             | No             |
| `capital-gain`   | Integer         | Capital gains                                                                                                                                                                                                            | No             |
| `capital-loss`   | Integer         | Capital losses                                                                                                                                                                                                           | No             |
| `hours-per-week` | Integer         | Number of hours worked per week                                                                                                                                                                                          | No             |
| `native-country` | Categorical     | Country of origin/residence, including United-States, India, Mexico, Canada, Germany, etc.                                                                                                                               | Yes            |
| `income`         | Binary / Target | `<=50K` or `>50K`                                                                                                                                                                                                        | No             |


### Numerical features

- age
- fnlwgt
- education-num
- capital-gain
- capital-loss
- hours-per-week

Numerical preprocessing uses:

- Median imputation
- StandardScaler

### Categorical features

- workclass
- education
- marital-status
- occupation
- relationship
- race
- sex
- native-country

Categorical preprocessing uses:

- Most-frequent imputation
- One-hot encoding
- `handle_unknown="ignore"`

The preprocessing and classifier are saved together as complete scikit-learn pipelines using Joblib.

## Github Repository Link

https://github.com/2025ac05048/ML-Assignment-2

### Github Repository Structure

```text
ML-Assignment-2/
│
├── app.py
├── train_models.py
├── requirements.txt
├── README.md
├── .gitignore
├── adult.data
├── adult.test
├── test_data.csv
│
├── model/
│   ├── logistic_regression.joblib
│   ├── decision_tree.joblib
│   ├── knn.joblib
│   ├── naive_bayes.joblib
│   ├── random_forest.joblib
│   └── preprocessor.joblib
│
└── results/
    └── model_comparison.csv
```

## Machine Learning Models

Five classification algorithms were implemented:

1. Logistic Regression
2. Decision Tree
3. K-Nearest Neighbors (KNN)
4. Gaussian Naive Bayes
5. Random Forest

## Model Performance

Evaluation was performed on the Adult test dataset containing 16,281 records.

| ML Model | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Random Forest | 0.8509 | 0.8995 | 0.7171 | 0.6089 | 0.6586 | 0.5671 |
| Logistic Regression | 0.8507 | 0.9038 | 0.7282 | 0.5871 | 0.6501 | 0.5618 |
| KNN | 0.8333 | 0.8564 | 0.6639 | 0.5962 | 0.6282 | 0.5224 |
| Decision Tree | 0.8127 | 0.7442 | 0.6013 | 0.6141 | 0.6077 | 0.4847 |
| Naive Bayes | 0.5511 | 0.7611 | 0.3378 | 0.9376 | 0.4967 | 0.3300 |

### Key observation

| ML Model Name                        | Observation about model performance                                                                                                                                                                                                                                                                                                                                                                                        |
| ------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Logistic Regression**              | Logistic Regression achieved **85.07% accuracy** and the **highest AUC (0.9038)** among all models, indicating strong overall classification and probability-ranking performance. It also achieved the **highest precision (0.7282)**, meaning that its predictions of the `>50K` class were relatively reliable. However, its recall of **0.5871** indicates that it missed a considerable number of actual `>50K` cases. |
| **Decision Tree**                    | Decision Tree achieved **81.27% accuracy**, which was the lowest among the tree-based and linear models except Naive Bayes. Its **AUC of 0.7442** was also the lowest, indicating weaker class-separation ability. Although its recall (**0.6141**) was slightly higher than Logistic Regression, its lower precision, F1 score (**0.6077**) and MCC (**0.4847**) indicate weaker overall performance.                     |
| **KNN**                              | KNN achieved **83.33% accuracy** with an AUC of **0.8564**. Its performance was better than Decision Tree and Naive Bayes on most overall metrics, but it remained below Logistic Regression and Random Forest. Its F1 score of **0.6282** indicates a reasonable balance between precision and recall, although it did not provide the best overall classification performance.                                           |
| **Naive Bayes**                      | Naive Bayes achieved a very high **recall of 0.9376**, meaning it identified most of the actual `>50K` cases. However, its **precision was only 0.3378**, resulting in many false-positive predictions. This led to a low accuracy (**55.11%**), F1 score (**0.4967**) and MCC (**0.3300**). Therefore, Naive Bayes prioritizes identifying positive cases at the expense of prediction reliability.                       |
| **Random Forest (Ensemble)**         | Random Forest achieved the **highest accuracy (85.09%)**, **highest recall (0.6089)**, **highest F1 score (0.6586)** and **highest MCC (0.5671)**. Its AUC (**0.8995**) was also very strong, although slightly below Logistic Regression. The combination of strong performance across multiple metrics makes Random Forest the most balanced model among the models tested.                                              |
| **Overall Winner for your dataset?** | **Random Forest.** It provides the best overall balance across the evaluation metrics, achieving the highest accuracy, recall, F1 score and MCC. Logistic Regression has a slightly higher AUC and precision, but Random Forest performs better overall and is therefore selected as the best-performing model for this dataset.                                                                                           |


## Streamlit Application

Live Streamlit App Link - https://2025ac05048.streamlit.app/

The Streamlit application provides:

- CSV test-data upload
- Model selection
- Prediction results
- Accuracy
- AUC
- Precision
- Recall
- F1 Score
- Matthews Correlation Coefficient (MCC)
- Confusion matrix
- Classification report
- Comparison of all five models
- F1-score comparison chart
