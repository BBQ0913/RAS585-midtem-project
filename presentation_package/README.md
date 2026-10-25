# Breast Cancer Classification Using Machine Learning

## Project Objective

Compare Logistic Regression, Decision Tree, k-Nearest Neighbors (kNN), and Support Vector Machine (SVM) to answer: "Which classification method achieves the highest breast cancer classification accuracy under this project's experimental setup?"

Use the Breast Cancer Wisconsin Diagnostic Dataset included in scikit-learn: 569 samples and 30 numerical features, with labels `0 = malignant` and `1 = benign`. No separate dataset download is needed.

## How to Run

Python 3.10 or newer is recommended. Run these commands in PowerShell from the project folder:

```powershell
# Create a separate Python environment for this project (only needed once).
python -m venv .venv
# Install the packages listed in requirements.txt into that environment.
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
# Run the full experiment using that same Python environment.
.\.venv\Scripts\python.exe main_code.py
```

If `.venv` already exists and the dependencies are installed, run only the last command.

All four methods are implemented: Logistic Regression with `C = 0.01, 0.1, 1, 10, 100`; Decision Tree with `max_depth = 3, 5, None`; kNN with `n_neighbors = 3, 5, 10`; and SVM with `linear` and `rbf` kernels at `C=1`. The RBF model uses `gamma="scale"`. Running the script compares 13 candidates using training-set cross-validation, selects one per method, and exports the results.

Each method also gets a parameter comparison CSV and chart. `decision_tree_depth5_confusion_matrix.png` describes the depth-5 candidate; `decision_tree_confusion_matrix.png` describes the CV-selected tree. Per-candidate test scores are for reporting only and never select parameters.

The `.vscode/settings.json` file points VS Code to the Windows `.venv` interpreter and includes English comments explaining each setting. `.gitignore` excludes the local environment, Python caches, and generated `results/` files from Git. The script uses Python's built-in `csv` module, so pandas is no longer required.

## Reading the Code as a Beginner

- A **sample** is one dataset row; a **feature** is one measured property. `X` contains the features and `y` contains the correct class labels.
- A **parameter setting** is a choice made before training, such as the number of neighbors. A **candidate** is a model created with one such setting.
- `build_*_models()` creates untrained candidates. These functions do not load data, train models, or write files.
- `prepare_data()` makes one shared split. `make_pipeline()` connects feature scaling to a fresh model.
- `evaluate_family()` compares candidates using CV, freezes the selected setting, then fits each candidate on the full training set for reporting.
- The `save_*()` functions create tables, images, and a combined summary. `main()` runs the workflow in order.

The settings near the top of `main_code.py` control the random seed, test fraction, number of CV folds, and output folder. English comments explain both these settings and the purpose of each processing step. Importing the module does not start the experiment; executing it directly does.

## Shared Workflow (Implemented)

1. Load the dataset and display the class labels.
2. Create an 80% / 20% stratified train/test split with `random_state = 42`. All four methods use the same split.
3. Place `StandardScaler` and the classifier in a single Pipeline. Scaling statistics are learned only from the training data used in each fit.
4. Compare each member's candidate configurations using the same five stratified cross-validation folds within the training set. Select the configuration with the highest mean validation accuracy; ties go to the first listed configuration.
5. Refit the selected configuration on the full training set, record training time, calculate training and test accuracy, and plot the test-set confusion matrix.
6. Refit the remaining candidates for descriptive reports, reusing their existing CV scores. These test scores cannot change the selected configuration.
7. Export parameter tables/charts, the final model table/chart, confusion matrix images, and a combined English summary.

Decision Tree does not require standardization, but uses the same Pipeline for consistency. Let the shared code train all models. Do not standardize the entire dataset beforehand or use the test set to select parameters.

## Member Responsibilities

In your assigned function in `main_code.py`, add the classifier import and fill in the `models` dictionary. Use the format `"descriptive configuration name": unfitted scikit-learn classifier object`, with one entry per parameter configuration. Do not call `fit` or `predict` inside these functions.

| Member | Function | Implementation and Comparison Tasks |
| --- | --- | --- |
| Member 1 | `build_logistic_regression_models()` | LogisticRegression; different C values; explain binary classification and regularization |
| Member 2 | `build_decision_tree_models()` | DecisionTreeClassifier; max_depth = 3, 5, None; analyze overfitting |
| Member 3 | `build_knn_models()` | KNeighborsClassifier; n_neighbors = 3, 5, 10; explain the effect of k |
| Member 4 | `build_svm_models()` | SVC; linear and rbf kernels; explain decision boundaries and combine the results |

Use the shared `RANDOM_STATE` for models with randomness. If a convergence warning appears, review the model settings instead of ignoring it. Member 3's review confirmed the requested k values, equal neighbor voting, Euclidean distance, and scaling inside each CV fit. Duplicate data splitting, CV, and reporting were removed from the builder functions. Decision Tree reporting also now uses the shared Pipeline, so its individual reports and final comparison follow the same workflow.

## Logistic Regression Results

Logistic Regression converts a linear score `z = w^T x + b` into the probability of class 1 using the sigmoid function `P(y=1 | x) = 1 / (1 + exp(-z))`. Here, class 1 is benign and class 0 is malignant. The default prediction is benign when this probability is greater than 0.5, and malignant otherwise.

The candidates use `max_iter=1000` and the default L2 regularization. Smaller C values apply stronger regularization; larger C values apply weaker regularization. Standardization is fitted within each training fold through the shared Pipeline.

| C | Mean Five-Fold CV Accuracy |
| --- | --- |
| 0.01 | 95.16% |
| **0.1** | **98.24%** |
| 1 | 97.80% |
| 10 | 96.26% |
| 100 | 95.38% |

Cross-validation selected `C=0.1`. After refitting on all 455 training samples, this model achieved **98.68% training accuracy** and **97.37% test accuracy** (111 of 114 test samples correct). No convergence warnings occurred in the verification run.

| Actual / Predicted | Malignant | Benign |
| --- | --- | --- |
| Malignant | 40 | 2 |
| Benign | 1 | 71 |

Two malignant samples were classified as benign, and one benign sample was classified as malignant. These results apply to the fixed split and CV settings above. The decrease in validation accuracy at larger C values suggests that weaker regularization does not improve generalization in this experiment.

## Decision Tree Results

A tree repeatedly splits samples using feature cutoffs. A larger `max_depth` allows more detailed decisions. `None` allows unlimited depth, subject to the classifier's other stopping rules.

| Maximum Depth | Mean Five-Fold CV Accuracy | Training Accuracy | Test Accuracy |
| --- | --- | --- | --- |
| **3** | **92.53%** | **97.58%** | **93.86%** |
| 5 | 92.31% | 99.34% | 92.11% |
| None | 91.65% | 100.00% | 91.23% |

CV selected depth 3. Increasing depth improves training accuracy while CV and test accuracy decrease in this run. This is evidence of overfitting: the deeper tree fits training details that do not transfer as well to new samples. The selected tree's confusion matrix is `[[38, 4], [3, 69]]`, with actual classes in rows and predicted classes in columns.

## kNN Results

kNN stores the training samples instead of learning a formula of weights. To classify a sample, it finds nearby training samples using Euclidean distance `d(x, x') = sqrt(sum_j (x_j - x'_j)^2)` and predicts the majority class among the k closest. Each neighbor has one vote. Because distance combines all 30 features, standardization inside the shared Pipeline is essential: without it, large-unit features such as area could dominate. At `k=10`, a 5-to-5 class vote is possible; scikit-learn chooses the smaller label, `0` (malignant), for a tied class vote.

A small k follows local structure closely but is sensitive to noisy neighbors; a larger k averages over more neighbors and gives a smoother, more stable decision boundary.

| k | Mean Five-Fold CV Accuracy | Training Accuracy | Test Accuracy |
| --- | --- | --- | --- |
| 3 | 96.26% | 97.80% | 98.25% |
| 5 | 96.26% | 97.36% | 95.61% |
| **10** | **96.92%** | **97.58%** | **96.49%** |

Cross-validation selected `k=10`. After refitting on all 455 training samples, this model achieved **97.58% training accuracy** and **96.49% test accuracy** (110 of 114 test samples correct).

| Actual / Predicted | Malignant | Benign |
| --- | --- | --- |
| Malignant | 39 | 3 |
| Benign | 1 | 71 |

Running the script also saves `knn_results.csv` (CV mean/std, training and test accuracy for every k) and `knn_accuracy.png` (those three accuracies plotted against k). `knn_confusion_matrix.png` describes the CV-selected model. The per-k test scores are for reporting, not parameter selection.

Three malignant samples were classified as benign, and one benign sample was classified as malignant. On the test set alone, k=3 would have scored 98.25%, but parameters are chosen by CV only; with 114 test samples, a one- or two-sample difference is not a reliable basis for choosing k.

## SVM Results (Member 4)

SVM finds a decision boundary that balances a wide margin with penalties for training errors or samples inside the margin. The closest influential training samples are called **support vectors**. A linear kernel creates a flat boundary in the feature space. An RBF (radial basis function) kernel uses similarity between samples to allow a curved boundary.

Both candidates use `C=1`, so this experiment compares kernel types while holding C fixed. Increasing C penalizes margin violations more strongly; decreasing C allows more violations in exchange for stronger regularization. RBF also uses `gamma="scale"`, calculated from the standardized training inputs. Larger gamma values make each sample's influence more local. The linear kernel does not use gamma.

| Kernel | Mean Five-Fold CV Accuracy | Training Accuracy | Test Accuracy |
| --- | --- | --- | --- |
| Linear | 96.48% | 99.12% | 97.37% |
| **RBF** | **96.92%** | **98.24%** | **98.25%** |

CV selected RBF. After refitting on all 455 training samples, it correctly classified **112 of 114 test samples**. Its slightly higher CV score supports choosing RBF between these two candidates, but the gap is small. The experiment does not establish that curved boundaries always perform better or that these are the best possible SVM settings.

| Actual / Predicted | Malignant | Benign |
| --- | --- | --- |
| Malignant | 41 | 1 |
| Benign | 1 | 71 |

One malignant sample was classified as benign and one benign sample as malignant. `svm_results.csv` and `svm_accuracy.png` compare the kernels; `svm_confusion_matrix.png` shows the selected RBF model.

## Outputs and Metrics

Outputs are saved to `results/` next to the script. A successful full run updates the current tables, charts, confusion matrices, and summary. Unrelated files from previous runs are not deleted. Accuracy results below were verified with Python 3.13.5, scikit-learn 1.9.1, NumPy 2.5.3, and Matplotlib 3.11.2; other supported package versions may produce small differences.

| File | Contents |
| --- | --- |
| `model_comparison.csv` | Status, best configuration, mean CV accuracy, training/test accuracy, training time in seconds, both error counts, and confusion matrix path for each method |
| `parameter_comparison.csv` | Mean CV accuracy and standard deviation for each candidate configuration |
| `*_results.csv` | Per-candidate CV mean/std, training/test accuracy, fit time, and `selected_by_cv` flag for each method |
| `*_accuracy.png` | Per-candidate training, CV, and test accuracy; CV error bars show one standard deviation across folds |
| `*_confusion_matrix.png` | Test-set confusion matrix for each method's selected configuration |
| `decision_tree_depth5_confusion_matrix.png` | Extra confusion matrix for the depth-5 tree candidate |
| `model_comparison.png` | All four selected models' CV/test accuracy and final fit time |
| `comparison_summary.txt` | Plain-English comparison, highest observed test accuracy, error counts, and limitations |

Accuracy values in the CSV files are fractions from 0 to 1; multiply by 100 for percentages. All four model statuses should be `completed`. Confusion matrix rows represent true classes and columns represent predicted classes, both ordered as malignant, benign. The upper-right cell counts malignant tumors incorrectly classified as benign. Standard deviation describes variation between the five fold scores; it is not a confidence interval.

All per-method CSV files now use the same column names, such as `setting`, `train_accuracy`, and `test_accuracy`. This replaces the older Decision Tree CSV's title-style column names. Existing output filenames are retained.

Training time is measured with `perf_counter()` around one final Pipeline `fit`. It includes standardization and model fitting, and excludes parameter search, prediction, and plotting. Timing depends on the computer and its current workload; a fast kNN fit does not necessarily imply fast prediction. Differences between training accuracy and CV/test accuracy can support the discussion of overfitting. Results describe performance under this particular data split and set of candidate configurations.

## Four-Method Comparison and Discussion (Member 4)

| Model | CV-Selected Setting | Mean CV Accuracy | Test Accuracy | Malignant Predicted Benign | Benign Predicted Malignant |
| --- | --- | --- | --- | --- | --- |
| Logistic Regression | C=0.1 | 98.24% | 97.37% | 2 | 1 |
| Decision Tree | max_depth=3 | 92.53% | 93.86% | 4 | 3 |
| kNN | k=10 | 96.92% | 96.49% | 3 | 1 |
| **SVM** | **RBF, C=1, gamma=scale** | **96.92%** | **98.25%** | **1** | **1** |

SVM has the highest observed test accuracy among the four CV-selected models in this experiment. Logistic Regression has the highest mean CV accuracy. These describe different evaluations: CV uses folds within the training set, while test accuracy uses the separate 114-sample test set. SVM makes only one fewer test error than Logistic Regression, so this does not prove a general advantage.

| Method | Advantages | Disadvantages and Observations |
| --- | --- | --- |
| Logistic Regression | Fast fit; probabilities; coefficients help explain the model | Linear boundary may miss nonlinear patterns; scaling and regularization matter |
| Decision Tree | Decisions can be expressed as rules; captures nonlinear patterns | Deep trees can overfit; unlimited depth reached 100% training accuracy but only 91.65% CV accuracy |
| kNN | Simple voting rule; little model-fitting work; flexible boundary | Stores training data; neighbor searches can be costly; sensitive to scaling and k |
| SVM | Margin-based classification; linear and nonlinear kernels | Scaling and parameter choices matter; RBF boundaries are harder to explain; training may become expensive on large datasets |

In the packaged results, final fit times were 0.002427 s (Logistic Regression), 0.002627 s (Decision Tree), 0.000803 s (kNN), 0.002025 s (SVM). These single measurements vary between runs and are not a robust speed ranking. Use results/model_comparison.csv as the numerical source of truth for this package. The chart displays milliseconds, while the CSV stores seconds.

For kNN, CV accuracy increased slightly from k=3/5 to k=10, consistent with a smoother boundary helping here. For SVM, RBF had slightly higher CV and test accuracy than linear at the chosen settings. Neither comparison establishes a universal rule.

The selected SVM misclassified one malignant sample as benign, compared with two for Logistic Regression, three for kNN, and four for the selected tree. Accuracy alone combines both types of error, so the confusion matrices provide useful additional detail.

The main limitations are a single fixed train/test split, a small test set, a limited set of candidate settings, and different numbers of candidates per method. We did not perform statistical significance tests or measure prediction speed. All conclusions apply to this dataset and workflow. Any future tuning should use training data and CV, with new held-out data reserved for a fresh final evaluation.

## Verification

Run the workflow checks with Python's built-in unittest tool:

```powershell
# Check builder behavior, scaling, CV-only selection, and complete output files.
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
# Optional: run the experiment and treat any warning as an error.
.\.venv\Scripts\python.exe -W error main_code.py
```

The integration check writes into a temporary folder, so it does not replace the results you use for your presentation.

## Seven-Slide Presentation Outline

1. Title: Project title, course, and team members.
2. Problem Description and Dataset: Research question, dataset features, and shared experimental workflow.
3. Logistic Regression: Method overview, C comparison, and individual results.
4. Decision Tree: Method overview, depth comparison, and overfitting observations.
5. k-Nearest Neighbors: Method overview, k comparison, and individual results.
6. Support Vector Machine: Method overview, linear/RBF comparison, and individual results.
7. Model Comparison and Conclusion: Accuracy, training time, advantages and disadvantages of each method, and conclusions.
