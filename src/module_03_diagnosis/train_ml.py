"""
第 3 层 - 故障诊断（传统机器学习基线）。

流程：CWRU 4 类信号 -> 滑窗分段 -> 组合特征 -> RF/SVM/KNN 分类 -> 评估。

运行：python src/module_03_diagnosis/train_ml.py
产出：
  figures/diagnosis/ml_confusion_matrix.png
  figures/diagnosis/ml_model_comparison.png
"""

import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.common.plot_style import set_style, savefig
from src.common.cwru_loader import load_dataset
from src.common.signal_utils import feature_vector

import matplotlib.pyplot as plt

CWRU_DIR = ROOT / "data" / "CWRU" / "CWRU_data"
FS = 12000.0
SEED = 42


def segment(record, win=2048, stride=1024, max_windows=40):
    x = record["signal"]
    rpm = record["rpm"]
    label = record["label"]
    out = []
    n = len(x)
    count = 0
    for start in range(0, n - win, stride):
        w = x[start:start + win]
        out.append((feature_vector(w, FS, rpm), label))
        count += 1
        if count >= max_windows:
            break
    return out


def main():
    set_style()
    print("加载 CWRU 数据集 ...")
    records = load_dataset(CWRU_DIR)
    print(f"样本(文件)数：{len(records)}")

    print("分段 + 提特征 ...")
    X, y = [], []
    for rec in records:
        for feat, lab in segment(rec):
            X.append(feat)
            y.append(lab)
    X = np.array(X)
    y = np.array(y)
    print(f"特征矩阵：{X.shape}，类别：{sorted(set(y))}")

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=SEED
    )

    models = {
        "RandomForest": make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=200, random_state=SEED)),
        "SVM-RBF": make_pipeline(StandardScaler(), SVC(kernel="rbf", C=10, gamma="scale")),
        "KNN-5": make_pipeline(StandardScaler(), KNeighborsClassifier(5)),
    }

    results = {}
    preds = {}
    for name, model in models.items():
        model.fit(X_tr, y_tr)
        p = model.predict(X_te)
        acc = accuracy_score(y_te, p)
        results[name] = acc
        preds[name] = p
        print(f"\n=== {name} 准确率：{acc:.4f} ===")
        print(classification_report(y_te, p, digits=4))

    # 文件级（跨文件/跨工况）评测：更贴近现场，体现泛化能力
    train_rec, test_rec = train_test_split(
        records, test_size=0.25, stratify=[r["label"] for r in records], random_state=SEED
    )

    def _build(rcs):
        Xf, yf = [], []
        for rec in rcs:
            for feat, lab in segment(rec):
                Xf.append(feat)
                yf.append(lab)
        return np.array(Xf), np.array(yf)

    Xf_tr, yf_tr = _build(train_rec)
    Xf_te, yf_te = _build(test_rec)
    rf_file = make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=200, random_state=SEED))
    rf_file.fit(Xf_tr, yf_tr)
    acc_file = accuracy_score(yf_te, rf_file.predict(Xf_te))
    print(f"\n[文件级划分] RandomForest 准确率：{acc_file:.4f}（跨文件/跨工况，更贴近现场）")

    # 混淆矩阵（选效果最好的模型展示）
    best = max(results, key=results.get)
    classes = sorted(set(y))
    cm = confusion_matrix(y_te, preds[best], labels=classes)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticklabels(classes)
    ax.set_xlabel("预测标签")
    ax.set_ylabel("真实标签")
    ax.set_title(f"混淆矩阵（{best}, acc={results[best]:.3f}）")
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="red" if i == j else "black", fontsize=11)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    savefig("ml_confusion_matrix", subdir="diagnosis")
    plt.close()

    # 模型对比
    fig, ax = plt.subplots(figsize=(7, 4))
    names = list(results.keys())
    vals = [results[n] for n in names]
    bars = ax.bar(names, vals, color=["#4C72B0", "#55A868", "#CCB974"])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("测试准确率")
    ax.set_title("传统 ML 故障诊断准确率对比")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}", ha="center")
    savefig("ml_model_comparison", subdir="diagnosis")
    plt.close()

    print(f"\n[完成] 最佳模型 {best}，准确率 {results[best]:.4f}")
    print("图已保存到 figures/diagnosis/")


if __name__ == "__main__":
    main()
