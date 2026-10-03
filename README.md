# Breast Cancer Classification Using Machine Learning

## 專案目標

比較 Logistic Regression、Decision Tree、k-Nearest Neighbors（kNN）與 Support Vector Machine（SVM），回答「在本專案的實驗設定下，哪一種分類方法有最高的乳癌分類準確率？」。

使用 scikit-learn 內建 Breast Cancer Wisconsin Diagnostic Dataset：569 筆資料、30 個數值特徵，標籤為 `0 = malignant（惡性）`、`1 = benign（良性）`，不需另外下載資料。

## 執行方式

建議使用 Python 3.10 或更新版本。在專案資料夾執行：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe main_code.py
```

若專案已有 `.venv` 且已安裝依賴，只需執行最後一行。

目前四個模型的實作都留白。直接執行會完成資料切分、顯示每個模型的 `pending` 狀態，並產生評估欄位留白的比較表。可先完成其中一個模型，其他模型仍會被跳過。

## 主體流程（已完成）

1. 載入資料與顯示類別對應。
2. 使用固定 `random_state = 42` 進行 80% / 20% 分層訓練／測試切分；四種方法共用同一份資料。
3. 將 `StandardScaler` 與分類器放入同一個 Pipeline。標準化只學習當次訓練資料的統計量。
4. 在訓練集內以相同的五折分層交叉驗證比較各組員提供的參數設定，取平均驗證準確率最高者；平手選先列出的設定。
5. 使用完整訓練集重新訓練選出的設定，記錄訓練時間，計算訓練與測試準確率，繪製測試集混淆矩陣。
6. 匯出參數比較表、四種方法的最終比較表與混淆矩陣圖片。

Decision Tree 不需要標準化，但為了共用流程，此處也透過 Pipeline 執行。所有模型必須由共用程式訓練，組員不要自行標準化整份資料或使用測試集挑參數。

## 組員填寫位置

在 `main_code.py` 各自的函式中加入分類器 import，並填入 `models` 字典。格式為 `"可辨識的設定名稱": 尚未訓練的 scikit-learn 分類器物件`，每種參數設定放一筆。不要在這些函式中呼叫 `fit` 或 `predict`。

| 組員 | 函式 | 待實作及比較項目 |
| --- | --- | --- |
| Member 1 | `build_logistic_regression_models()` | LogisticRegression；不同 C；二元分類及正則化說明 |
| Member 2 | `build_decision_tree_models()` | DecisionTreeClassifier；max_depth = 3、5、None；過擬合分析 |
| Member 3 | `build_knn_models()` | KNeighborsClassifier；n_neighbors = 3、5、10；k 值影響 |
| Member 4 | `build_svm_models()` | SVC；linear、rbf kernel；決策邊界說明及整合結果 |

有隨機性的模型請使用共用 `RANDOM_STATE`。若出現收斂警告，請確認各自模型設定；不要直接忽略警告。函式內也保留了原理與實驗觀察的文字欄位。

## 輸出與指標

輸出放在程式所在資料夾下的 `results/`，CSV 每次執行會更新，混淆矩陣則只對本次已實作的模型產生／更新。以本次 CSV 的狀態與圖片路徑為準；資料夾可能保留先前執行的圖片。

| 檔案 | 內容 |
| --- | --- |
| `model_comparison.csv` | 每種方法的狀態、最佳設定、CV 平均準確率、訓練／測試準確率、訓練秒數、混淆矩陣路徑 |
| `parameter_comparison.csv` | 每個候選設定的 CV 平均準確率與標準差 |
| `*_confusion_matrix.png` | 每種方法選定設定的測試集混淆矩陣 |

CSV 的準確率為 0–1 小數，簡報中請轉為百分比；尚未實作的模型保留空白，不能當成 0%。混淆矩陣的列是真實類別、欄是預測類別，順序均為 malignant、benign；右上角表示惡性被誤判為良性。

訓練時間以 `perf_counter()` 計算一次最終 Pipeline 的 `fit`，包含標準化與模型訓練，不包含參數搜尋、預測或繪圖。時間受電腦與當時負載影響；kNN 的 fit 較快，也不代表它的預測一定較快。訓練準確率與 CV／測試準確率的落差可輔助討論過擬合。比較結果僅描述這次資料切分與候選設定下的表現。

## 最終比較與討論（組員完成實驗後填寫）

| Model | Test Accuracy | Training Time (s) |
| --- | --- | --- |
| Logistic Regression | | |
| Decision Tree | | |
| kNN | | |
| SVM | | |

- 最高測試準確率的方法：
- Logistic Regression 的優缺點：
- Decision Tree 的優缺點及深度比較：
- kNN 的優缺點及 k 值比較：
- SVM 的優缺點及 kernel 比較：
- 混淆矩陣觀察（尤其是惡性被誤判為良性）：
- 結論與限制：

## 七頁簡報架構

1. Title：題目、課程、組員。
2. Problem Description and Dataset：研究問題、資料特徵、共用實驗流程。
3. Logistic Regression：原理、C 比較、個人結果。
4. Decision Tree：原理、深度比較、過擬合觀察。
5. k-Nearest Neighbors：原理、k 值比較、個人結果。
6. Support Vector Machine：原理、linear／RBF 比較、個人結果。
7. Model Comparison and Conclusion：準確率、訓練時間、各方法優缺點與結論。
