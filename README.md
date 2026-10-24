# Breast Cancer Classification Using Machine Learning

## Project Objective

Compare Logistic Regression, Decision Tree, k-Nearest Neighbors (kNN), and Support Vector Machine (SVM) to answer: "Which classification method achieves the highest breast cancer classification accuracy under this project's experimental setup?"

Use the Breast Cancer Wisconsin Diagnostic Dataset included in scikit-learn: 569 samples and 30 numerical features, with labels `0 = malignant` and `1 = benign`. No separate dataset download is needed.

## How to Run

Python 3.10 or newer is recommended. Run these commands in PowerShell from the project folder:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe main_code.py
```

If `.venv` already exists and the dependencies are installed, run only the last command.

Logistic Regression is implemented with `C = 0.01, 0.1, 1, 10, 100`, Decision Tree is implemented with `max_depth = 3, 5, None`, and kNN is implemented with `n_neighbors = 3, 5, 10`. Running the script compares each method's candidates using training-set cross-validation, evaluates the selected model, and exports its results. SVM is still blank and displays `pending` with blank evaluation fields.

The current Decision Tree function also runs a separate depth comparison and saves `decision_tree_results.csv`, `decision_tree_accuracy.png`, and `decision_tree_depth5_confusion_matrix.png`. The depth-5 plot describes that specific candidate; `decision_tree_confusion_matrix.png` describes the model selected by cross-validation. The per-depth test scores are for reporting, not parameter selection.

## Shared Workflow (Implemented)

1. Load the dataset and display the class labels.
2. Create an 80% / 20% stratified train/test split with `random_state = 42`. All four methods use the same split.
3. Place `StandardScaler` and the classifier in a single Pipeline. Scaling statistics are learned only from the training data used in each fit.
4. Compare each member's candidate configurations using the same five stratified cross-validation folds within the training set. Select the configuration with the highest mean validation accuracy; ties go to the first listed configuration.
5. Refit the selected configuration on the full training set, record training time, calculate training and test accuracy, and plot the test-set confusion matrix.
6. Export the parameter comparison table, final model comparison table, and confusion matrix images.

Decision Tree does not require standardization, but uses the same Pipeline for consistency. Let the shared code train all models. Do not standardize the entire dataset beforehand or use the test set to select parameters.

## Where Each Member Adds Their Work

In your assigned function in `main_code.py`, add the classifier import and fill in the `models` dictionary. Use the format `"descriptive configuration name": unfitted scikit-learn classifier object`, with one entry per parameter configuration. Do not call `fit` or `predict` inside these functions.

| Member | Function | Implementation and Comparison Tasks |
| --- | --- | --- |
| Member 1 | `build_logistic_regression_models()` | LogisticRegression; different C values; explain binary classification and regularization |
| Member 2 | `build_decision_tree_models()` | DecisionTreeClassifier; max_depth = 3, 5, None; analyze overfitting |
| Member 3 | `build_knn_models()` | KNeighborsClassifier; n_neighbors = 3, 5, 10; explain the effect of k |
| Member 4 | `build_svm_models()` | SVC; linear and rbf kernels; explain decision boundaries and combine the results |

Use the shared `RANDOM_STATE` for models with randomness. If a convergence warning appears, review your model settings instead of ignoring it. The unfinished functions include blank fields for explaining the method and recording experimental observations.

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

## kNN Results

kNN stores the training samples and learns no parameters. To classify a sample, it computes the Euclidean distance `d(x, x') = sqrt(sum_j (x_j - x'_j)^2)` to every training sample and predicts the majority class among the k closest. Because the distance combines all 30 features, standardization inside the shared Pipeline is essential: without it, large-unit features such as area would dominate.

A small k follows local structure closely but is sensitive to noisy neighbors; a larger k averages over more neighbors and gives a smoother, more stable decision boundary.

| k | Mean Five-Fold CV Accuracy |
| --- | --- |
| 3 | 96.26% |
| 5 | 96.26% |
| **10** | **96.92%** |

Cross-validation selected `k=10`. After refitting on all 455 training samples, this model achieved **97.58% training accuracy** and **96.49% test accuracy** (110 of 114 test samples correct).

| Actual / Predicted | Malignant | Benign |
| --- | --- | --- |
| Malignant | 39 | 3 |
| Benign | 1 | 71 |

Running the script also saves `knn_results.csv` (CV mean/std, training and test accuracy for every k) and `knn_accuracy.png` (those three accuracies plotted against k). `knn_confusion_matrix.png` describes the CV-selected model. The per-k test scores are for reporting, not parameter selection.

Three malignant samples were classified as benign, and one benign sample was classified as malignant. On the test set alone, k=3 would have scored 98.25%, but parameters are chosen by CV only; with 114 test samples, a one- or two-sample difference is not a reliable basis for choosing k.

## Outputs and Metrics

Outputs are saved to `results/` next to the script. CSV files are updated on each run; confusion matrix images are generated or updated only for models implemented in that run. Use the current CSV status and image paths as your reference, since images from previous runs may remain in the folder.

| File | Contents |
| --- | --- |
| `model_comparison.csv` | Status, best configuration, mean CV accuracy, training/test accuracy, training time in seconds, and confusion matrix path for each method |
| `parameter_comparison.csv` | Mean CV accuracy and standard deviation for each candidate configuration |
| `*_confusion_matrix.png` | Test-set confusion matrix for each method's selected configuration |

Accuracy values in the CSV files are fractions from 0 to 1; convert them to percentages for the presentation. Unfinished models have blank metrics, which must not be interpreted as 0%. Confusion matrix rows represent true classes and columns represent predicted classes, both ordered as malignant, benign. The upper-right cell counts malignant tumors incorrectly classified as benign.

Training time is measured with `perf_counter()` around one final Pipeline `fit`. It includes standardization and model fitting, and excludes parameter search, prediction, and plotting. Timing depends on the computer and its current workload; a fast kNN fit does not necessarily imply fast prediction. Differences between training accuracy and CV/test accuracy can support the discussion of overfitting. Results describe performance under this particular data split and set of candidate configurations.

## Current Comparison and Discussion (Complete After All Methods Are Implemented)

| Model | Test Accuracy | Training Time (s) |
| --- | --- | --- |
| Logistic Regression | 97.37% | 0.002191 |
| Decision Tree | 93.86% | 0.002730 |
| kNN | 96.49% | 0.001097 |
| SVM | pending | pending |

The values above reflect the latest local verification run; training times vary between runs. See `results/model_comparison.csv` for the most recently generated measurements.

- Method with the highest test accuracy: Logistic Regression among the three implemented methods. The four-method comparison remains incomplete until SVM is implemented.
- Logistic Regression advantages and disadvantages: Fast to train and provides class probabilities and interpretable coefficients in standardized feature units. Its linear decision boundary may miss nonlinear relationships, and its performance depends on regularization and feature scaling.
- Decision Tree advantages, disadvantages, and depth comparison:
- kNN advantages, disadvantages, and comparison of k values: Simple, has no training cost beyond storing the data, and makes no assumption about the shape of the decision boundary. Prediction is slow on large datasets because every query is compared with all training samples, and results depend heavily on feature scaling and the choice of k. CV accuracy rose slightly from k=3/5 (96.26%) to k=10 (96.92%), suggesting the smoother boundary generalizes a little better here.
- SVM advantages, disadvantages, and kernel comparison:
- Confusion matrix observations, especially malignant tumors classified as benign:
- Conclusions and limitations:

## Seven-Slide Presentation Outline

1. Title: Project title, course, and team members.
2. Problem Description and Dataset: Research question, dataset features, and shared experimental workflow.
3. Logistic Regression: Method overview, C comparison, and individual results.
4. Decision Tree: Method overview, depth comparison, and overfitting observations.
5. k-Nearest Neighbors: Method overview, k comparison, and individual results.
6. Support Vector Machine: Method overview, linear/RBF comparison, and individual results.
7. Model Comparison and Conclusion: Accuracy, training time, advantages and disadvantages of each method, and conclusions.
