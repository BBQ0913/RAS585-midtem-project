"""Breast Cancer Classification Using Machine Learning.

四位組員只需填寫各自的 build_*_models()，回傳尚未訓練的分類器。
空字典代表尚未完成；不會產生虛構的模型或評估結果。
"""

import csv
from pathlib import Path
from time import perf_counter

import matplotlib

matplotlib.use("Agg")  # 將圖存成檔案，支援沒有視窗的執行環境。
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
    """TODO：加入 LogisticRegression 分類器，比較不同 C。

    請在此函式內加入需要的 import。
    回傳格式：{"參數設定名稱": 尚未 fit 的分類器物件, ...}
    說明欄位（由組員填寫）：
      二元分類原理：
      C 與正則化的關係：
      實驗觀察：
    """
    models = {
        # TODO: 填入 Logistic Regression 的參數設定及模型。
    }
    return models


# =============================================================================
# Member 2 — Decision Tree
# =============================================================================
def build_decision_tree_models():
    """TODO：加入 DecisionTreeClassifier，比較 max_depth = 3、5、None。

    請在此函式內加入需要的 import，並使用 RANDOM_STATE。
    回傳格式：{"參數設定名稱": 尚未 fit 的分類器物件, ...}
    說明欄位（由組員填寫）：
      決策樹原理：
      模型深度與過擬合的關係：
      實驗觀察：
    """
    models = {
        # TODO: 填入 Decision Tree 的參數設定及模型。
    }
    return models


# =============================================================================
# Member 3 — k-Nearest Neighbors
# =============================================================================
def build_knn_models():
    """TODO：加入 KNeighborsClassifier，比較 k = 3、5、10。

    請在此函式內加入需要的 import。
    回傳格式：{"參數設定名稱": 尚未 fit 的分類器物件, ...}
    說明欄位（由組員填寫）：
      kNN 原理與距離計算：
      k 值對分類結果的影響：
      實驗觀察：
    """
    models = {
        # TODO: 填入 kNN 的參數設定及模型。
    }
    return models


# =============================================================================
# Member 4 — Support Vector Machine
# =============================================================================
def build_svm_models():
    """TODO：加入 SVC，比較 linear 與 rbf kernel。

    請在此函式內加入需要的 import。
    回傳格式：{"參數設定名稱": 尚未 fit 的分類器物件, ...}
    說明欄位（由組員填寫）：
      決策邊界與最大間隔：
      Linear 與 RBF kernel 的差異：
      實驗觀察與四種方法的總結：
    """
    models = {
        # TODO: 填入 SVM 的參數設定及模型。
    }
    return models


MODEL_BUILDERS = {
    "Logistic Regression": build_logistic_regression_models,
    "Decision Tree": build_decision_tree_models,
    "kNN": build_knn_models,
    "SVM": build_svm_models,
}


def prepare_data():
    """固定分層切分；標準化留在 Pipeline 內以避免資料洩漏。"""
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
    """每次 fit 僅從該次訓練資料學習平均值與標準差。"""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", clone(estimator)),
    ])


def evaluate_family(name, candidates, X_train, X_test, y_train, y_test, class_names):
    """用 CV 挑選該方法的設定，再進行一次最終測試集評估。"""
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

    # 平手時選字典中先出現的設定，不使用測試集打破平手。
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
