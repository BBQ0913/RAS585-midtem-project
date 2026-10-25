# Presentation Brief

Prepared from the existing project outputs on 2026-10-25 (America/Phoenix).
The experiment was not rerun during packaging, preserving the supplied measurements.

## Purpose and dataset

Research question: Which of the four classification methods has the highest test accuracy under this experimental setup?

- Course: RAS585.
- Dataset: Breast Cancer Wisconsin Diagnostic, bundled with scikit-learn.
- 569 samples; 30 numerical features; labels 0 = malignant and 1 = benign.
- Training: 455 samples. Test: 114 samples (42 malignant, 72 benign).
- Stratified 80/20 split, seed 42; five stratified CV folds within training data.
- StandardScaler and classifier in one Pipeline; no full-dataset scaling.
- 13 candidates; choose one per method by mean CV accuracy, ties to the first listed.

## Selected models: use these in the final comparison

| Model | Setting | CV accuracy | Training accuracy | Test accuracy | Fit time (s) | Malignant as benign | Benign as malignant |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | C=0.1 | 98.24% | 98.68% | 97.37% | 0.002427 | 2 | 1 |
| Decision Tree | max_depth=3 | 92.53% | 97.58% | 93.86% | 0.002627 | 4 | 3 |
| kNN | k=10 | 96.92% | 97.58% | 96.49% | 0.000803 | 3 | 1 |
| SVM | kernel=rbf, C=1, gamma=scale | 96.92% | 98.24% | 98.25% | 0.002025 | 1 | 1 |

## Parameter comparisons

Bold settings were chosen using CV. Test scores did not determine these choices.

### Logistic Regression

| Setting | Mean CV accuracy | CV standard deviation | Training accuracy | Test accuracy |
| --- | --- | --- | --- | --- |
| C=0.01 | 95.16% | 1.49 percentage points | 95.60% | 94.74% |
| **C=0.1** | 98.24% | 0.54 percentage points | 98.68% | 97.37% |
| C=1 | 97.80% | 0.98 percentage points | 98.90% | 98.25% |
| C=10 | 96.26% | 1.12 percentage points | 99.12% | 96.49% |
| C=100 | 95.38% | 1.62 percentage points | 99.34% | 94.74% |

### Decision Tree

| Setting | Mean CV accuracy | CV standard deviation | Training accuracy | Test accuracy |
| --- | --- | --- | --- | --- |
| **max_depth=3** | 92.53% | 0.82 percentage points | 97.58% | 93.86% |
| max_depth=5 | 92.31% | 1.20 percentage points | 99.34% | 92.11% |
| max_depth=None | 91.65% | 1.79 percentage points | 100.00% | 91.23% |

### kNN

| Setting | Mean CV accuracy | CV standard deviation | Training accuracy | Test accuracy |
| --- | --- | --- | --- | --- |
| k=3 | 96.26% | 0.88 percentage points | 97.80% | 98.25% |
| k=5 | 96.26% | 1.12 percentage points | 97.36% | 95.61% |
| **k=10** | 96.92% | 1.28 percentage points | 97.58% | 96.49% |

### SVM

| Setting | Mean CV accuracy | CV standard deviation | Training accuracy | Test accuracy |
| --- | --- | --- | --- | --- |
| kernel=linear, C=1 | 96.48% | 1.62 percentage points | 99.12% | 97.37% |
| **kernel=rbf, C=1, gamma=scale** | 96.92% | 1.46 percentage points | 98.24% | 98.25% |

## Suggested slide plan and existing figures

| Slide | Topic | Suggested figure | Key message |
| --- | --- | --- | --- |
| 1 | Title and team | None needed | Use placeholders for unknown names |
| 2 | Dataset and workflow | Create a simple workflow diagram from the stated steps | Split, scale within CV, select, refit, evaluate |
| 3 | Logistic Regression | results/logistic_regression_accuracy.png | C=0.1 selected; stronger/weaker regularization |
| 4 | Decision Tree | results/decision_tree_accuracy.png | Depth 3 selected; deeper trees overfit here |
| 5 | kNN | results/knn_accuracy.png | k=10 selected by CV; distance requires scaling |
| 6 | SVM | results/svm_accuracy.png; optionally results/svm_confusion_matrix.png | RBF selected over linear at C=1 |
| 7 | Comparison and conclusion | results/model_comparison.png | SVM leads on this test split; Logistic Regression leads on CV |

All four selected-model confusion matrices are also included. The extra decision_tree_depth5_confusion_matrix.png is for an unselected candidate and must not replace the selected tree matrix in the final comparison.

## Key explanations and conclusions

- Logistic Regression uses a linear score to estimate a class probability. Smaller C means stronger regularization. It is easy to interpret but may miss nonlinear structure.
- A Decision Tree asks a sequence of feature-cutoff questions. Unlimited depth reaches 100% training accuracy but lower CV accuracy than depth 3, suggesting overfitting.
- kNN finds nearby training samples and lets them vote. Small k follows local details; large k smooths the boundary. Equal votes and Euclidean distance are used.
- SVM balances a wide margin against margin violations. Linear uses a flat boundary; RBF allows a curved boundary. Both use C=1; RBF uses gamma=scale.
- Selected SVM makes 2 test errors; Logistic Regression 3; kNN 4; Decision Tree 7.
- SVM is the highest observed test scorer, not a proven universally best model. Its advantage over Logistic Regression is just one test sample.
- CV error bars show one standard deviation across folds, not confidence intervals.
- Only one split and limited settings were evaluated. No significance tests or prediction-speed benchmark were performed. Training times vary across runs.

## Source priority

Use results/model_comparison.csv for final numbers, results/*_results.csv for candidate tables, main_code.py for implementation, and README.md for explanations. The packaged README timing paragraph was synchronized to these CSVs.

## Running the supplied code (optional)

The dataset ships with scikit-learn; no separate raw-data download is required. See README.md for setup commands. Rerunning the code changes measured timings. Use the included output snapshot for this presentation.

The package includes five workflow tests in tests/test_workflow.py. They passed during the preceding code verification; packaging checks confirm four completed models, 13 CV candidates, matching selected-model CSV fields, and archive integrity.
