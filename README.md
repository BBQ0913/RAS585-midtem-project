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

All four model implementations are currently blank. Running the script splits the data, displays `pending` for each model, and exports a comparison table with blank evaluation fields. You can implement one model at a time; unfinished models will be skipped.

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

Use the shared `RANDOM_STATE` for models with randomness. If a convergence warning appears, review your model settings instead of ignoring it. Each function also includes blank fields for explaining the method and recording experimental observations.

## Outputs and Metrics

Outputs are saved to `results/` next to the script. CSV files are updated on each run; confusion matrix images are generated or updated only for models implemented in that run. Use the current CSV status and image paths as your reference, since images from previous runs may remain in the folder.

| File | Contents |
| --- | --- |
| `model_comparison.csv` | Status, best configuration, mean CV accuracy, training/test accuracy, training time in seconds, and confusion matrix path for each method |
| `parameter_comparison.csv` | Mean CV accuracy and standard deviation for each candidate configuration |
| `*_confusion_matrix.png` | Test-set confusion matrix for each method's selected configuration |

Accuracy values in the CSV files are fractions from 0 to 1; convert them to percentages for the presentation. Unfinished models have blank metrics, which must not be interpreted as 0%. Confusion matrix rows represent true classes and columns represent predicted classes, both ordered as malignant, benign. The upper-right cell counts malignant tumors incorrectly classified as benign.

Training time is measured with `perf_counter()` around one final Pipeline `fit`. It includes standardization and model fitting, and excludes parameter search, prediction, and plotting. Timing depends on the computer and its current workload; a fast kNN fit does not necessarily imply fast prediction. Differences between training accuracy and CV/test accuracy can support the discussion of overfitting. Results describe performance under this particular data split and set of candidate configurations.

## Final Comparison and Discussion (Complete After Running the Experiments)

| Model | Test Accuracy | Training Time (s) |
| --- | --- | --- |
| Logistic Regression | | |
| Decision Tree | | |
| kNN | | |
| SVM | | |

- Method with the highest test accuracy:
- Logistic Regression advantages and disadvantages:
- Decision Tree advantages, disadvantages, and depth comparison:
- kNN advantages, disadvantages, and comparison of k values:
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
