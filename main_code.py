"""Compare four classifiers on the same breast cancer dataset.

Read from top to bottom: imports, settings, each member's models, shared helpers,
and main(), which runs the experiment. X means the input measurements; y means
known class labels. See README.md for measured results and their interpretation.
"""

# These tools come with Python; no separate installation is needed.
import csv  # Write tables that Excel or a text editor can open.
from pathlib import Path  # Build file paths on different computers.
from time import perf_counter  # Measure elapsed time in seconds.

# Install external packages with: python -m pip install -r requirements.txt
import matplotlib

# Choose the backend BEFORE importing pyplot. Agg saves plots without opening
# windows, so this script also works on computers without a display.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np  # Calculate averages and standard deviations.
from sklearn.base import clone  # Make a fresh, untrained copy of a model.
from sklearn.datasets import load_breast_cancer  # Load the bundled dataset.
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline  # Keep scaling and training together.
from sklearn.preprocessing import StandardScaler  # Put features on similar scales.


# =============================================================================
# Shared settings: change these here instead of changing each member's code.
# =============================================================================
# A fixed seed repeats the same random split and folds each time we run the code.
RANDOM_STATE = 42
# Reserve 20% of samples for testing; use the other 80% for training.
TEST_SIZE = 0.2
# Cross-validation (CV) splits training data into five parts. Each part takes a
# turn as validation data, while the other four parts are used to train a model.
CV_FOLDS = 5
# __file__ is this script's location. Save results beside it even when the command
# is run from another folder. main() creates the folder if it does not exist.
OUTPUT_DIR = Path(__file__).resolve().parent / "results"


# =============================================================================
# Member 1: Yi-Hung Ting - Logistic Regression
# =============================================================================
def build_logistic_regression_models():
    """Create untrained Logistic Regression candidates.

    Binary classification means choosing between two classes. Logistic Regression
    combines inputs into a score z = w^T x + b, then uses the sigmoid function
    1 / (1 + exp(-z)) to estimate the probability of class 1 (benign).
    The default prediction is benign above 0.5, and malignant otherwise.

    Regularization discourages overly large model weights. Smaller C means
    stronger regularization; larger C lets the model fit training data more closely.
    """
    from sklearn.linear_model import LogisticRegression

    # A dictionary connects each readable setting name to its model object.
    # max_iter allows up to 1,000 optimization steps to find the model weights.
    models = {
        "C=0.01": LogisticRegression(C=0.01, max_iter=1000),
        "C=0.1": LogisticRegression(C=0.1, max_iter=1000),
        "C=1": LogisticRegression(C=1.0, max_iter=1000),
        "C=10": LogisticRegression(C=10.0, max_iter=1000),
        "C=100": LogisticRegression(C=100.0, max_iter=1000),
    }
    # Building a model does not train it. Shared code below performs every fit.
    return models


# =============================================================================
# Member 2 - Decision Tree
# =============================================================================
def build_decision_tree_models():
    """Create trees with different limits on the number of decision levels.

    A tree asks a sequence of questions, such as whether a measurement is above
    a cutoff. A leaf (the end of a path) gives the predicted class. max_depth
    limits the number of levels; None means no depth limit.
    A deeper tree can memorize training details instead of learning patterns
    that work on new samples. This problem is called overfitting.
    """
    from sklearn.tree import DecisionTreeClassifier

    # Use the same seed for all depths so random choices can be reproduced.
    # Like the other builders, this function does not load data or train models.
    return {
        "max_depth=3": DecisionTreeClassifier(max_depth=3, random_state=RANDOM_STATE),
        "max_depth=5": DecisionTreeClassifier(max_depth=5, random_state=RANDOM_STATE),
        "max_depth=None": DecisionTreeClassifier(max_depth=None, random_state=RANDOM_STATE),
    }


# =============================================================================
# Member 3 - k-Nearest Neighbors (kNN)
# =============================================================================
def build_knn_models():
    """Create kNN candidates that ask 3, 5, or 10 nearby samples to vote.

    kNN keeps training samples rather than learning a formula of weights.
    For a new sample, it finds the k closest training samples and predicts the
    most common class among them. Every neighbor has one vote (uniform weights).
    Euclidean distance is the straight-line distance across the features:
    d(x, x') = sqrt(sum_j (x_j - x'_j)^2).

    Scaling is essential: otherwise a feature with large numbers, such as area,
    could dominate a feature with small numbers, such as smoothness. The shared
    Pipeline learns scaling only from the data used for training.

    A small k reacts strongly to nearby details, including noise. A larger k
    averages more votes and gives a smoother boundary, but can miss local patterns.
    For k=10, a 5-to-5 vote is possible; scikit-learn chooses the smaller class
    label (0, malignant) in a tied class vote. We select k using CV, never by
    choosing the largest test score. README.md explains the measured comparison.
    """
    from sklearn.neighbors import KNeighborsClassifier

    # Explicit options make the distance and voting rules easy to find.
    return {
        f"k={k}": KNeighborsClassifier(
            n_neighbors=k, weights="uniform", metric="euclidean",
        )
        for k in (3, 5, 10)
    }


# =============================================================================
# Member 4 - Support Vector Machine (SVM)
# =============================================================================
def build_svm_models():
    """Create SVM candidates with linear and RBF decision boundaries.

    A decision boundary separates the regions predicted as the two classes.
    SVM seeks a wide margin: a gap between the boundary and nearby samples.
    The samples that determine this boundary are called support vectors.
    C balances a wider margin against penalties for errors or margin violations.

    A linear kernel produces a straight boundary in two dimensions (a flat
    separating surface in our 30-feature space). An RBF, or radial basis function,
    kernel uses similarity between samples to allow a curved boundary. gamma
    controls how local that similarity is: larger values focus on closer points.

    Hold C=1.0 fixed to compare the kernels. gamma='scale' calculates gamma from
    training inputs for RBF; the linear kernel does not use it. These are two
    chosen configurations, not an exhaustive SVM parameter search.
    Shared code selects the kernel by CV and combines all four methods' results.
    """
    from sklearn.svm import SVC

    # SVC means Support Vector Classifier. Both candidates receive standardized
    # features through make_pipeline(), just like kNN.
    return {
        "kernel=linear, C=1": SVC(kernel="linear", C=1.0),
        "kernel=rbf, C=1, gamma=scale": SVC(kernel="rbf", C=1.0, gamma="scale"),
    }


# Store function names WITHOUT parentheses: main() calls each function later.
# Dictionary order also controls the order of methods in the output tables.
MODEL_BUILDERS = {
    "Logistic Regression": build_logistic_regression_models,
    "Decision Tree": build_decision_tree_models,
    "kNN": build_knn_models,
    "SVM": build_svm_models,
}


# =============================================================================
# Shared data preparation and evaluation
# =============================================================================
def prepare_data():
    """Load data once and return the same train/test split for every method."""
    dataset = load_breast_cancer()
    # Each data row is one sample; each column is one numerical feature.
    # stratify keeps the proportions of the two classes similar in both sets.
    X_train, X_test, y_train, y_test = train_test_split(
        dataset.data, dataset.target,
        test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=dataset.target,
    )
    print(f"Dataset: {dataset.data.shape[0]} samples, {dataset.data.shape[1]} features")
    print(f"Train: {len(y_train)} | Test: {len(y_test)}")
    print("Class labels: 0 = malignant, 1 = benign\n")
    return X_train, X_test, y_train, y_test, dataset.target_names


def make_pipeline(estimator):
    """Create a fresh scaler and model that are trained together, in that order."""
    # StandardScaler subtracts each feature's training mean and divides by its
    # training standard deviation. Validation/test data reuse those values.
    # Keeping it INSIDE the Pipeline prevents validation/test information from
    # leaking into training. clone() leaves the original candidate untrained.
    # Trees do not need scaling, but all four methods use the same steps here.
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", clone(estimator)),
    ])


def evaluate_family(name, candidates, X_train, X_test, y_train, y_test, class_names):
    """Choose a setting by training-set CV, then save its evaluation and reports."""
    if not candidates:
        raise ValueError(f"{name} must provide at least one candidate model.")

    # Reusing the seed gives each method the same five stratified folds.
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    validation_rows = []
    for setting, estimator in candidates.items():
        # Each fold fits a fresh Pipeline. Test data are never passed into CV.
        # n_jobs=1 runs one job at a time; error_score='raise' exposes fit failures.
        scores = cross_val_score(
            make_pipeline(estimator), X_train, y_train,
            scoring="accuracy", cv=cv, n_jobs=1, error_score="raise",
        )
        validation_rows.append({
            "model": name,
            "setting": setting,
            "cv_accuracy_mean": float(np.mean(scores)),  # Average of five scores.
            "cv_accuracy_std": float(np.std(scores)),  # Variation across folds.
        })

    # Freeze the choice BEFORE looking at any test score. For tied means, max()
    # returns the first listed candidate. Test accuracy never breaks the tie.
    best = max(validation_rows, key=lambda row: row["cv_accuracy_mean"])
    comparison_rows = []
    for validation in validation_rows:
        setting = validation["setting"]
        fitted = make_pipeline(candidates[setting])
        # Time only this full-training-set fit, including scaling. CV, prediction,
        # and plotting are outside the timer. Timings vary between computers/runs.
        start = perf_counter()
        fitted.fit(X_train, y_train)
        training_seconds = perf_counter() - start
        predictions = fitted.predict(X_test)
        row = {
            **validation,  # Copy the CV fields into this more detailed row.
            "train_accuracy": accuracy_score(y_train, fitted.predict(X_train)),
            "test_accuracy": accuracy_score(y_test, predictions),
            "training_seconds": training_seconds,
            "selected_by_cv": setting == best["setting"],
        }
        comparison_rows.append(row)
        # Rows are actual classes, columns are predictions, in order [0, 1].
        matrix = confusion_matrix(y_test, predictions, labels=[0, 1])

        if row["selected_by_cv"]:
            image_path = OUTPUT_DIR / f"{file_stem(name)}_confusion_matrix.png"
            save_confusion_matrix(matrix, class_names, f"{name}\n{setting}", image_path)
            result = {
                "model": name,
                "status": "completed",
                "best_setting": setting,
                "cv_accuracy_mean": row["cv_accuracy_mean"],
                "train_accuracy": row["train_accuracy"],
                "test_accuracy": row["test_accuracy"],
                "training_seconds": training_seconds,
                "malignant_as_benign": int(matrix[0, 1]),
                "benign_as_malignant": int(matrix[1, 0]),
                "confusion_matrix_file": image_path.name,
            }
            print(f"{name}: {setting}")
            print(f"  CV accuracy: {row['cv_accuracy_mean']:.2%}")
            print(f"  Test accuracy: {row['test_accuracy']:.2%}")
            print(f"  Training time: {training_seconds:.6f} s")
            print(f"  Confusion matrix (rows=true, columns=predicted):\n{matrix}\n")

        # Keep the existing Member 2 depth-5 report as a separate candidate plot.
        if name == "Decision Tree" and setting == "max_depth=5":
            save_confusion_matrix(
                matrix, class_names, "Decision Tree (max_depth=5)",
                OUTPUT_DIR / "decision_tree_depth5_confusion_matrix.png",
            )

    # Other candidates' test scores are descriptive only: they cannot change
    # best or result. CV is calculated once and reused in these reports.
    save_parameter_comparison(name, comparison_rows)
    return result, validation_rows


# =============================================================================
# Shared output helpers
# =============================================================================
def file_stem(name):
    """Turn a display name such as 'Decision Tree' into 'decision_tree'."""
    return name.lower().replace(" ", "_")


def write_csv(path, rows, fieldnames):
    """Write one dictionary per row; fieldnames sets the column order."""
    # 'w' replaces the old table. newline='' avoids extra blank lines on Windows.
    # utf-8-sig helps Excel recognize the text encoding correctly.
    with path.open("w", newline="", encoding="utf-8-sig") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_confusion_matrix(matrix, class_names, title, path):
    """Save correct and incorrect prediction counts as a labeled image."""
    # Diagonal cells are correct. The upper-right cell counts malignant samples
    # predicted as benign; the lower-left cell counts the opposite mistake.
    figure, axis = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(matrix, display_labels=class_names).plot(
        ax=axis, cmap="Blues", colorbar=False,
    )
    axis.set_title(title)
    figure.tight_layout()  # Leave room for labels so they are not cut off.
    figure.savefig(path, dpi=160)
    plt.close(figure)  # Release memory after saving the figure.


def save_parameter_comparison(name, rows):
    """Save each candidate's CV, training, and test scores for the presentation."""
    stem = file_stem(name)
    write_csv(OUTPUT_DIR / f"{stem}_results.csv", rows, list(rows[0]))
    # Place each setting at an integer position along the horizontal axis.
    positions = np.arange(len(rows))
    # Break long SVM labels onto multiple lines so they remain readable.
    labels = [row["setting"].replace(", ", "\n") for row in rows]
    figure, axis = plt.subplots(figsize=(9, 6))
    axis.plot(positions, [row["train_accuracy"] for row in rows], marker="o",
              label="Training accuracy")
    # Error bars show one standard deviation across folds, not a confidence range.
    axis.errorbar(positions, [row["cv_accuracy_mean"] for row in rows],
                  yerr=[row["cv_accuracy_std"] for row in rows], marker="D",
                  capsize=4, label=f"{CV_FOLDS}-fold CV mean +/- std")
    axis.plot(positions, [row["test_accuracy"] for row in rows], marker="s",
              label="Test accuracy (report only)")
    selected = next(i for i, row in enumerate(rows) if row["selected_by_cv"])
    labels[selected] += "\n(selected by CV)"
    axis.set_xticks(positions, labels)
    axis.set_xlabel("Candidate setting")
    axis.set_ylabel("Accuracy")
    # Scores are stored as fractions; display percentages in plots.
    axis.yaxis.set_major_formatter(PercentFormatter(1.0))
    axis.set_title(f"{name}: parameter comparison")
    axis.grid(True, alpha=0.3)
    axis.legend(loc="best")
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / f"{stem}_accuracy.png", dpi=160)
    plt.close(figure)


def save_overall_comparison(results):
    """Combine all four selected models into a chart and an English summary."""
    labels = [row["model"].replace(" ", "\n") for row in results]
    positions = np.arange(len(results))
    figure, (accuracy_axis, time_axis) = plt.subplots(1, 2, figsize=(12, 6))
    # Compare the CV-selected models, not each method's highest-scoring test result.
    accuracy_axis.bar(positions - 0.2, [r["cv_accuracy_mean"] for r in results],
                      width=0.4, label="CV accuracy")
    test_bars = accuracy_axis.bar(
        positions + 0.2, [r["test_accuracy"] for r in results],
        width=0.4, label="Test accuracy",
    )
    accuracy_axis.bar_label(test_bars, labels=[f"{r['test_accuracy']:.2%}" for r in results],
                            padding=3, fontsize=9)
    accuracy_axis.set_ylim(0, 1.12)  # Leave room for labels above the bars.
    accuracy_axis.yaxis.set_major_formatter(PercentFormatter(1.0))
    accuracy_axis.set_ylabel("Accuracy")
    accuracy_axis.set_title("Accuracy of each selected model")
    accuracy_axis.legend(loc="lower right")
    # Convert seconds to milliseconds so short fit times are easier to read.
    time_bars = time_axis.bar(positions, [r["training_seconds"] * 1000 for r in results])
    time_axis.bar_label(time_bars, fmt="%.2f", padding=3)
    time_axis.margins(y=0.2)
    time_axis.set_ylabel("Final fit time (milliseconds)")
    time_axis.set_title("Training time (varies between runs)")
    for axis in (accuracy_axis, time_axis):
        axis.set_xticks(positions, labels)
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "model_comparison.png", dpi=160)
    plt.close(figure)

    # Report every tied winner, instead of naming only the first one.
    highest_accuracy = max(row["test_accuracy"] for row in results)
    winners = [r["model"] for r in results if r["test_accuracy"] == highest_accuracy]
    lines = [
        "Comparison of four classifiers",
        "Each method's setting was chosen using training-set cross-validation.",
        f"Highest observed test accuracy: {', '.join(winners)} ({highest_accuracy:.2%}).",
        "",
    ]
    for row in results:
        lines.append(
            f"{row['model']} ({row['best_setting']}): "
            f"CV {row['cv_accuracy_mean']:.2%}, "
            f"training {row['train_accuracy']:.2%}, test {row['test_accuracy']:.2%}; "
            f"final fit {row['training_seconds']:.6f} seconds; "
            f"malignant predicted benign: {row['malignant_as_benign']}; "
            f"benign predicted malignant: {row['benign_as_malignant']}."
        )
    lines.extend([
        "",
        "These results describe one fixed split and the listed candidate settings.",
        "Test scores compare the chosen models; they must not be used to retune them.",
        "A small accuracy difference does not establish a universally better method.",
        "Fit time excludes prediction; a fast kNN fit does not imply fast prediction.",
    ])
    summary = "\n".join(lines) + "\n"
    (OUTPUT_DIR / "comparison_summary.txt").write_text(summary, encoding="utf-8")
    print(summary)


def main():
    """Run the experiment and write tables, plots, and a combined summary."""
    X_train, X_test, y_train, y_test, class_names = prepare_data()
    # exist_ok=True lets the script run again if the folder already exists.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = []  # One row for each member's selected model.
    validation_results = []  # One row for every candidate's CV score.

    for name, builder in MODEL_BUILDERS.items():
        candidates = builder()
        result, validation_rows = evaluate_family(
            name, candidates, X_train, X_test, y_train, y_test, class_names,
        )
        results.append(result)  # append adds a single row.
        validation_results.extend(validation_rows)  # extend adds several rows.

    write_csv(OUTPUT_DIR / "model_comparison.csv", results, [
        "model", "status", "best_setting", "cv_accuracy_mean", "train_accuracy",
        "test_accuracy", "training_seconds", "malignant_as_benign",
        "benign_as_malignant", "confusion_matrix_file",
    ])
    write_csv(OUTPUT_DIR / "parameter_comparison.csv", validation_results, [
        "model", "setting", "cv_accuracy_mean", "cv_accuracy_std",
    ])
    save_overall_comparison(results)
    print(f"Results saved to: {OUTPUT_DIR}")
    print("Accuracy values in CSV files are fractions from 0 to 1 (0.95 means 95%).")


# Run only when this file is executed directly. Importing it makes the functions
# available to other files without starting the experiment or writing results.
if __name__ == "__main__":
    main()
