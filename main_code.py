"""Breast Cancer Classification Using Machine Learning.
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
# Yi-Hung Ting — Logistic Regression
# =============================================================================
def build_logistic_regression_models():
    """Build Logistic Regression models with different C values.
    How binary classification works:
      Logistic Regression maps a linear score z = w^T x + b to a probability
      with the sigmoid function: P(y=1 | x) = 1 / (1 + exp(-z)).
      Here, class 0 is malignant and class 1 is benign. The default prediction
      is benign when P(y=1 | x) > 0.5, and malignant otherwise.
    Relationship between C and regularization:
      A smaller C gives stronger regularization, which makes the model simpler.
      A larger C gives weaker regularization, which allows the model to fit
      the training data more closely.
    Experimental observations:
      Five C values were tested: 0.01, 0.1, 1, 10, and 100.
      C=0.1 had the highest cross-validation accuracy about 98.24%.
      The final model achieved about 98.68% training accuracy and 97.37%
      test accuracy. Its test confusion matrix was [[40, 2], [1, 71]],
      with rows=true and columns=predicted, ordered malignant, benign.
      These observations use the shared split and five-fold CV settings.
    """
    from sklearn.linear_model import LogisticRegression
    models = {
        "C=0.01": LogisticRegression(C=0.01, max_iter=1000),
        "C=0.1": LogisticRegression(C=0.1, max_iter=1000),
        "C=1": LogisticRegression(C=1.0, max_iter=1000),
        "C=10": LogisticRegression(C=10.0, max_iter=1000),
        "C=100": LogisticRegression(C=100.0, max_iter=1000),
    }
    return models



# =============================================================================
# Member 2 — Decision Tree
# =============================================================================
def build_decision_tree_models():
    """Decision Tree classification and depth comparison."""
    
    import pandas as pd
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import accuracy_score, confusion_matrix

    # Also allow this function to be run independently.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load dataset
    data = load_breast_cancer()

    X = data.data
    y = data.target

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    print("\n========== Decision Tree ==========")
    print("Training samples:", len(X_train))
    print("Testing samples:", len(X_test))

    depths = [3, 5, None]

    results = []
    trained_models = {}
    models = {}

    # Compare different maximum depths
    for depth in depths:

        model = DecisionTreeClassifier(
            max_depth=depth,
            random_state=RANDOM_STATE
        )

        start = perf_counter()

        model.fit(X_train, y_train)

        training_time = perf_counter() - start

        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)

        train_acc = accuracy_score(y_train, train_pred)
        test_acc = accuracy_score(y_test, test_pred)

        results.append({
            "Max Depth": str(depth),
            "Training Accuracy": train_acc,
            "Testing Accuracy": test_acc,
            "Training Time (s)": training_time
        })

        trained_models[str(depth)] = model

        # Return unfitted classifiers to the main program
        models[f"max_depth={depth}"] = DecisionTreeClassifier(
            max_depth=depth,
            random_state=RANDOM_STATE
        )

    df = pd.DataFrame(results)

    print(df.to_string(index=False))

    # Save results
    df.to_csv(
        OUTPUT_DIR / "decision_tree_results.csv",
        index=False
    )

    # Accuracy comparison
    plt.figure(figsize=(8, 5))

    plt.plot(
        df["Max Depth"],
        df["Training Accuracy"],
        marker="o",
        label="Training Accuracy"
    )

    plt.plot(
        df["Max Depth"],
        df["Testing Accuracy"],
        marker="s",
        label="Testing Accuracy"
    )

    plt.xlabel("Max Depth")
    plt.ylabel("Accuracy")
    plt.title("Decision Tree Accuracy Comparison")
    plt.legend()
    plt.grid(True)

    plt.savefig(
        OUTPUT_DIR / "decision_tree_accuracy.png",
        dpi=300
    )
    plt.close()

    # Confusion Matrix (Max Depth = 5)
    model = trained_models["5"]

    y_pred = model.predict(X_test)

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=[0, 1]
    )

    fig, ax = plt.subplots(figsize=(6, 5))

    ConfusionMatrixDisplay(
        cm,
        display_labels=["Malignant", "Benign"]
    ).plot(
        ax=ax,
        cmap="Blues",
        colorbar=False
    )

    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Decision Tree Confusion Matrix (Depth = 5)")

    plt.savefig(
        OUTPUT_DIR / "decision_tree_depth5_confusion_matrix.png",
        dpi=300
    )
    plt.close()

    return models



# =============================================================================
# Member 3 — k-Nearest Neighbors
# =============================================================================
def build_knn_models():
    """Build kNN models with different k values.
    How kNN works and how distances are calculated:
      kNN stores the training samples and learns no parameters. To classify a
      new sample, it computes the Euclidean distance to every training sample,
      d(x, x') = sqrt(sum_j (x_j - x'_j)^2), takes the k closest neighbors,
      and predicts the majority class among them. Because distance mixes all
      30 features, the shared Pipeline standardizes them first so features
      with large units (e.g. area) do not dominate small ones (e.g. smoothness).
    Effect of k on classification results:
      A small k follows local structure closely, so it has low bias but is
      sensitive to noisy or mislabeled neighbors (higher variance).
      A larger k averages over more neighbors, giving smoother decision
      boundaries that are more stable but may blur the class border.
    Experimental observations:
      Three k values were tested: 3, 5, and 10.
      k=3 and k=5 both reached about 96.26% cross-validation accuracy, and
      k=10 had the highest at about 96.92%, so the smoother boundary helped.
      The final model achieved about 97.58% training accuracy and 96.49%
      test accuracy. Its test confusion matrix was [[39, 3], [1, 71]],
      with rows=true and columns=predicted, ordered malignant, benign.
      On the test set alone, k=3 scored highest (98.25%), then k=10 (96.49%)
      and k=5 (95.61%). k=10 is still the selected model because the choice
      is made by CV; 114 test samples are too few to pick k reliably.
      These observations use the shared split and five-fold CV settings.
    """
    from sklearn.neighbors import KNeighborsClassifier
    models = {
        "k=3": KNeighborsClassifier(n_neighbors=3),
        "k=5": KNeighborsClassifier(n_neighbors=5),
        "k=10": KNeighborsClassifier(n_neighbors=10),
    }
    plot_knn_comparison(models)
    return models


def plot_knn_comparison(models):
    """Save per-k train/test accuracy for reporting; selection still uses CV only."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y,
    )
    rows = []
    for setting, estimator in models.items():
        # kNN is distance-based, so scale inside the Pipeline as in the shared workflow.
        fitted = make_pipeline(estimator).fit(X_train, y_train)
        rows.append({
            "setting": setting,
            "train_accuracy": accuracy_score(y_train, fitted.predict(X_train)),
            "test_accuracy": accuracy_score(y_test, fitted.predict(X_test)),
        })
    write_csv(OUTPUT_DIR / "knn_results.csv", rows,
              ["setting", "train_accuracy", "test_accuracy"])

    labels = [row["setting"] for row in rows]
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(labels, [row["train_accuracy"] for row in rows], marker="o", label="Training Accuracy")
    axis.plot(labels, [row["test_accuracy"] for row in rows], marker="s", label="Testing Accuracy")
    axis.set_xlabel("Number of Neighbors (k)")
    axis.set_ylabel("Accuracy")
    axis.set_title("kNN Accuracy Comparison")
    axis.legend()
    axis.grid(True)
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "knn_accuracy.png", dpi=300)
    plt.close(figure)


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
