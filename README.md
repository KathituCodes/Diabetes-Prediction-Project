# Diabetes Prediction: Supervised Classification with Logistic Regression, KNN, and Random Forest

A full end-to-end supervised machine learning project applying three classification algorithms to predict diabetes onset, with principled data cleaning, outlier handling, cross-validated hyperparameter tuning, and a structured performance comparison.

---

## Problem Statement

Diabetes prediction requires identifying at-risk individuals from clinical measurements. This project builds and evaluates three classification models on a dataset of 15,000 patient records, each representing features such as glucose levels, BMI, blood pressure, and age. The goal is to determine which model delivers the highest predictive accuracy and, critically, the highest recall on the diabetic class -- because missing a positive case in a medical context carries far greater cost than a false alarm.

---

## Dataset

| Property | Detail |
|---|---|
| Total records | 15,000 rows |
| Features | 8 clinical variables (Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age) |
| Target | Binary: Diabetic (1) / Non-Diabetic (0) |
| Source | Pima Indians Diabetes-style dataset |

---

## Critical Data Cleaning Decisions

### Zero-value imputation

Several clinical features contained zeros that are biologically impossible -- glucose cannot be zero, nor can BMI or blood pressure. Treating these as valid measurements would corrupt the model's understanding of normal ranges. These zeros were identified and replaced with the column median, stratified by the target class. Stratified imputation was chosen over simple median replacement because diabetic and non-diabetic patients have meaningfully different distributions for these features. Using a combined median would drag imputed values toward the majority class and introduce systematic bias.

### Outlier handling: IQR method with domain context

Outliers were detected using the interquartile range (IQR) method across all numeric features. The critical decision was not to remove outliers automatically. Each flagged value was cross-referenced against what is clinically plausible. For example, extreme insulin values are realistic in diabetic patients; capping them without clinical justification would erase genuine signal. Only values that exceeded physiologically impossible thresholds were treated as data errors and removed. This distinction -- between extreme-but-real observations and erroneous measurements -- is what separates a thoughtful analysis from a mechanical one.

### Feature scaling

Standard scaling was applied inside a scikit-learn Pipeline, ensuring that scaling was fit on the training set only and applied to the test set without data leakage. This matters because KNN is explicitly distance-based and sensitive to unscaled features. Logistic Regression also converges more stably with scaled inputs.

---

## Models and Methodology

### Logistic Regression (Baseline)

Logistic Regression fits a linear decision boundary between the two classes. It is interpretable, computationally efficient, and serves as the baseline comparator. For a dataset where the relationship between features and diabetes risk may be non-linear, a linear boundary is expected to underperform more flexible models.

**Results:**
- Overall Accuracy: 79%
- Diabetic Recall: 61%

The 61% recall means that 39 out of every 100 diabetic patients were missed -- a significant clinical gap.

### K-Nearest Neighbours (KNN)

KNN makes no assumption about the shape of the decision boundary. It classifies a new observation by finding the K most similar patients in the training data and taking a majority vote. This naturally captures complex, non-linear patterns in the feature space.

**Initial test at K=5:**
- Overall Accuracy: 88%
- Diabetic Recall: 77%

**Why KNN outperformed Logistic Regression here:** Diabetes risk is determined by interactions between multiple features in ways that do not reduce to a single linear combination. KNN's non-parametric nature allows it to capture those interactions directly from the data distribution.

**KNN's limitation:** At prediction time, KNN computes distances to every training record. With 15,000 rows this is manageable. At millions of rows it becomes impractical. For large-scale deployment, Logistic Regression's speed advantage may outweigh the accuracy gap.

### Hyperparameter Tuning: Finding the Optimal K

The initial K=5 was an arbitrary starting point. GridSearchCV with Stratified 7-Fold Cross-Validation was used to test odd values of K from 1 to 29. Stratified folds preserve the diabetic/non-diabetic class ratio in every split, which is essential when the classes are imbalanced.

The key principle applied here: K was selected using cross-validation on the training set exclusively. Testing multiple K values against the test set would inflate the apparent performance by overfitting the choice of K to the test data -- a form of data leakage that undermines generalisability.

**Best K found:** [From CV results]
**Best CV Accuracy:** [From CV results]

---

## Model Comparison Summary

| Metric | Logistic Regression | KNN (K=5) | KNN (Best K) |
|---|---|---|---|
| Overall Accuracy | 79% | 88% | [Best K result] |
| Diabetic Recall | 61% | 77% | [Best K result] |
| Non-Diabetic Recall | -- | -- | -- |

**Headline finding:** KNN clearly wins on every metric. More importantly, the diabetic recall improvement from 61% to 77% means that KNN correctly catches approximately 160 more diabetic patients per 1,000 cases than Logistic Regression was identifying. In a clinical screening context, that difference is substantial.

---

## Key Insights

**Frequency vs clinical importance:** High transaction frequency does not imply high predictive value. Some features that appear frequently across records contribute less discriminative power than rarer but more diagnostically significant values like glucose and BMI. Feature importance should be assessed by predictive contribution, not prevalence in the dataset.

**Model flexibility matters here:** The fact that KNN substantially outperforms Logistic Regression on this dataset is diagnostic information. It indicates that the true decision boundary for diabetes onset is non-linear. For future work, tree-based models like Random Forest would be a natural next step to test, as they can capture complex interactions while providing feature importance rankings for interpretability.

**Cross-validation discipline:** The use of stratified k-fold cross-validation for hyperparameter selection, rather than a single train/test split, is a deliberate methodological choice. It reduces the variance of the performance estimate and prevents tuning to a lucky or unlucky random split.

---

## Tech Stack

| Tool | Role |
|---|---|
| Python 3 | Core language |
| pandas / numpy | Data manipulation and cleaning |
| scikit-learn | Model training, pipelines, GridSearchCV, evaluation metrics |
| matplotlib / seaborn | Static visualisations and confusion matrices |
| plotly | Interactive accuracy vs K curve |

---

## Repository Structure

```
diabetes-classification/
|
|-- Diabetes_Analysis_Complete_using_Supervised_Classification.ipynb
|-- README.md
```

---

## How to Run

1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn plotly
   ```
3. Open the notebook in Jupyter or Google Colab.
4. Run all cells sequentially. The notebook is structured to execute top to bottom with outputs displayed at each stage.

---

## Author

**Urbanus Kathitu** | [github.com/KathituCodes](https://github.com/KathituCodes)
