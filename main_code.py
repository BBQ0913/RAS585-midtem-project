"""Breast Cancer Classification Using Machine Learning.
test
Each team member fills in their own build_*_models() with unfitted classifiers.
An empty dictionary marks unfinished work; no models or results are fabricated.
"""

import csv
from pathlib import Path
from time import perf_counter

import matplotlib

matplotlib.use("Agg")  # Save plots to files, including in environments without a GUI.
import matplotlib.pyplot as plt
import numpy as np
from sklearn.base import clone
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
OUTPUT_DIR = Path(__file__).resolve().parent / "results"


# =============================================================================
# Member 1 — Logistic Regression
# =============================================================================
def build_logistic_regression_models():
    """TODO: Add LogisticRegression classifiers and compare different C values.

    Add the required imports inside this function.
    Return format: {"configuration name": unfitted classifier object, ...}
    Explanation fields (to be completed by the assigned member):
      How binary classification works:
      Relationship between C and regularization:
      Experimental observations:
    """
    models = {
        # TODO: Add Logistic Regression configurations and model objects.
    }
    return models


# =============================================================================
# Member 2 — Decision Tree
# =============================================================================
def build_decision_tree_models():
    """TODO: Add DecisionTreeClassifier models with max_depth = 3, 5, and None.

    Add the required imports inside this function and use RANDOM_STATE.
    Return format: {"configuration name": unfitted classifier object, ...}
    Explanation fields (to be completed by the assigned member):
      How decision trees work:
      Relationship between tree depth and overfitting:
      Experimental observations:
    """
    models = {
        # TODO: Add Decision Tree configurations and model objects.
    }
    return models


# =============================================================================
# Member 3 — k-Nearest Neighbors
# =============================================================================
def build_knn_models():
    """TODO: Add KNeighborsClassifier models with k = 3, 5, and 10.

    Add the required imports inside this function.
    Return format: {"configuration name": unfitted classifier object, ...}
    Explanation fields (to be completed by the assigned member):
      How kNN works and how distances are calculated:
      Effect of k on classification results:
      Experimental observations:
    """
    models = {
        # TODO: Add kNN configurations and model objects.
    }
    return models


# =============================================================================
# Member 4 — Support Vector Machine
# =============================================================================
def build_svm_models():
    """TODO: Add SVC models and compare linear and rbf kernels.

    Add the required imports inside this function.
    Return format: {"configuration name": unfitted classifier object, ...}
    Explanation fields (to be completed by the assigned member):
      Decision boundaries and maximum margin:
      Differences between linear and RBF kernels:
      Experimental observations and a summary of all four methods:
    """
    models = {
        # TODO: Add SVM configurations and model objects.
    }
    return models


MODEL_BUILDERS = {
    "Logistic Regression": build_logistic_regression_models,
    "Decision Tree": build_decision_tree_models,
    "kNN": build_knn_models,
    "SVM": build_svm_models,
}


def prepare_data():
    """Use a fixed stratified split; keep scaling in the Pipeline to avoid leakage."""
    dataset = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        dataset.data,
        dataset.target,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=dataset.target,
    )
    print(f"Dataset: {dataset.data.shape[0]} samples, {dataset.data.shape[1]} features")
    print(f"Train: {len(y_train)} | Test: {len(y_test)}")
    print("Class labels: 0 = malignant, 1 = benign\n")
    return X_train, X_test, y_train, y_test, dataset.target_names


def make_pipeline(estimator):
    """Learn scaling means and standard deviations only from each fit's training data."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", clone(estimator)),
    ])


def evaluate_family(name, candidates, X_train, X_test, y_train, y_test, class_names):
    """Select a configuration using CV, then evaluate it once on the test set."""
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    validation_rows = []
    for setting, estimator in candidates.items():
        scores = cross_val_score(
            make_pipeline(estimator), X_train, y_train,
            scoring="accuracy", cv=cv, n_jobs=1, error_score="raise",
        )
        validation_rows.append({
            "model": name,
            "setting": setting,
            "cv_accuracy_mean": float(np.mean(scores)),
            "cv_accuracy_std": float(np.std(scores)),
        })

    # Break ties by dictionary insertion order, without consulting the test set.
    best = max(validation_rows, key=lambda row: row["cv_accuracy_mean"])
    fitted = make_pipeline(candidates[best["setting"]])
    start = perf_counter()
    fitted.fit(X_train, y_train)
    training_seconds = perf_counter() - start
    predictions = fitted.predict(X_test)
    matrix = confusion_matrix(y_test, predictions, labels=[0, 1])

    figure, axis = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(matrix, display_labels=class_names).plot(
        ax=axis, cmap="Blues", colorbar=False,
    )
    axis.set_title(f"{name}\n{best['setting']}")
    figure.tight_layout()
    image_path = OUTPUT_DIR / f"{name.lower().replace(' ', '_')}_confusion_matrix.png"
    figure.savefig(image_path, dpi=160)
    plt.close(figure)

    result = {
        "model": name,
        "status": "completed",
        "best_setting": best["setting"],
        "cv_accuracy_mean": best["cv_accuracy_mean"],
        "train_accuracy": accuracy_score(y_train, fitted.predict(X_train)),
        "test_accuracy": accuracy_score(y_test, predictions),
        "training_seconds": training_seconds,
        "confusion_matrix_file": image_path.name,
    }
    print(f"{name}: {best['setting']}")
    print(f"  Test accuracy: {result['test_accuracy']:.2%}")
    print(f"  Training time: {training_seconds:.6f} s")
    print(f"  Confusion matrix (rows=true, columns=predicted):\n{matrix}\n")
    return result, validation_rows


def write_csv(path, rows, fieldnames):
    with path.open("w", newline="", encoding="utf-8-sig") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    X_train, X_test, y_train, y_test, class_names = prepare_data()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    validation_results = []

    for name, builder in MODEL_BUILDERS.items():
        candidates = builder()
        if not candidates:
            print(f"{name}: pending (member implementation required)")
            results.append({"model": name, "status": "pending"})
            continue
        result, validation_rows = evaluate_family(
            name, candidates, X_train, X_test, y_train, y_test, class_names,
        )
        results.append(result)
        validation_results.extend(validation_rows)

    write_csv(OUTPUT_DIR / "model_comparison.csv", results, [
        "model", "status", "best_setting", "cv_accuracy_mean", "train_accuracy",
        "test_accuracy", "training_seconds", "confusion_matrix_file",
    ])
    write_csv(OUTPUT_DIR / "parameter_comparison.csv", validation_results, [
        "model", "setting", "cv_accuracy_mean", "cv_accuracy_std",
    ])
    print(f"\nResults saved to: {OUTPUT_DIR}")
    print("Pending models have blank metrics. Accuracy values in CSV are fractions (0–1).")


if __name__ == "__main__":
    main()
