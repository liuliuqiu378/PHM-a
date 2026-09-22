"""
场景 B：PHM 2024 北美挑战赛 —— 涡桨发动机扭矩裕度预测 + 故障检测
=========================================================================

任务本质（来自官方 PDF 的归纳）：
- 输入是发动机的多传感器「快照」：trq_measured(实测扭矩)、oat(外界温度)、
  mgt(燃气温度?)、pa(气压高度)、ias(指示空速)、np(螺旋桨转速%)、ng(核心机转速%)。
- 输出有两个：
    * trq_margin：扭矩裕度（连续值，越大越健康，为负表示已故障）—— 这是「健康指标/RUL 替代量」
    * faulty：是否故障（0/1 标签）
- 这是一个「用工况 + 传感器读数预测发动机健康」的真实回归 + 分类问题。
  它比 CWRU 轴承分类更接近工程现场：特征间强相关、存在工况耦合、需要回归而非硬分类。

本脚本做：数据探查(EDA) → 多模型基线对比 → 预测可视化 → 特征重要性，
全部结果落盘到 figures/scenario_engine/，并打印关键指标。

运行：从 PHM 根目录执行
  python src/scenarios/scenario_engine/train.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.common.plot_style import set_style, savefig
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.metrics import (
    r2_score, mean_squared_error, roc_auc_score, accuracy_score,
)

DATA_DIR = ROOT / "data" / "PHM2024" / "extracted"
FEATURES = ["trq_measured", "oat", "mgt", "pa", "ias", "np", "ng"]
SEED = 42


def load() -> pd.DataFrame:
    X = pd.read_csv(DATA_DIR / "training" / "X_train.csv")
    y = pd.read_csv(DATA_DIR / "training" / "y_train.csv")
    df = X.merge(y, on="id")
    return df


def eda(df: pd.DataFrame):
    set_style()
    # 1) 特征与扭矩裕度的相关系数热图
    corr = df[FEATURES + ["trq_margin"]].corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticklabels(corr.columns)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center",
                    fontsize=9, color="black")
    ax.set_title("特征与扭矩裕度的相关性")
    fig.colorbar(im, fraction=0.046, pad=0.04)
    savefig("correlation", "scenario_engine")

    # 2) 故障样本 vs 正常样本的扭矩裕度分布
    fig, ax = plt.subplots()
    for lab, color, name in [(0, "#2ca02c", "正常"), (1, "#d62728", "故障")]:
        vals = df.loc[df["faulty"] == lab, "trq_margin"]
        ax.hist(vals, bins=60, alpha=0.6, color=color, label=name, density=True)
    ax.axvline(0, color="k", ls="--", lw=1)
    ax.set_xlabel("扭矩裕度 trq_margin")
    ax.set_ylabel("密度")
    ax.set_title("正常 / 故障样本的扭矩裕度分布")
    ax.legend()
    savefig("margin_dist", "scenario_engine")

    faulty_rate = df["faulty"].mean()
    print(f"[EDA] 样本数 {len(df)}，故障率 {faulty_rate:.3f}")
    print("[EDA] 与 trq_margin 相关性最强的特征：")
    print(corr["trq_margin"].drop("trq_margin").abs().sort_values(ascending=False).round(3).to_string())


def build_models():
    regressors = {
        "Ridge(线性)": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "RandomForest": RandomForestRegressor(n_estimators=300, n_jobs=-1, random_state=SEED),
        "MLP": make_pipeline(
            StandardScaler(),
            MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=400,
                         early_stopping=True, random_state=SEED),
        ),
    }
    classifiers = {
        "RandomForest": RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=SEED),
        "MLP": make_pipeline(
            StandardScaler(),
            MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=400,
                          early_stopping=True, random_state=SEED),
        ),
    }
    return regressors, classifiers


def main():
    set_style()
    df = load()
    eda(df)

    X = df[FEATURES].values
    y_margin = df["trq_margin"].values
    y_fault = df["faulty"].values

    X_tr, X_te, ym_tr, ym_te, yf_tr, yf_te = train_test_split(
        X, y_margin, y_fault, test_size=0.2, random_state=SEED, stratify=y_fault
    )

    # ---- 回归：预测扭矩裕度 ----
    regressors, classifiers = build_models()
    print("\n=== 回归任务：预测 trq_margin（健康指标）===")
    reg_results = {}
    best_name, best_r2 = None, -1e9
    for name, model in regressors.items():
        model.fit(X_tr, ym_tr)
        pred = model.predict(X_te)
        rmse = float(np.sqrt(mean_squared_error(ym_te, pred)))
        r2 = float(r2_score(ym_te, pred))
        reg_results[name] = (rmse, r2, pred)
        print(f"  {name:14s} RMSE={rmse:8.3f}  R²={r2:.4f}")
        if r2 > best_r2:
            best_name, best_r2 = name, r2

    # 回归模型对比条形图
    fig, ax = plt.subplots()
    names = list(reg_results.keys())
    r2s = [reg_results[n][1] for n in names]
    ax.bar(names, r2s, color="#1f77b4")
    for i, v in enumerate(r2s):
        ax.text(i, v + 0.005, f"{v:.3f}", ha="center", fontsize=10)
    ax.set_ylim(0, max(r2s) * 1.15)
    ax.set_ylabel("R²")
    ax.set_title("各模型对扭矩裕度的解释力 (R²)")
    savefig("regression_compare", "scenario_engine")

    # 最佳模型的预测 vs 真实
    best_pred = reg_results[best_name][2]
    fig, ax = plt.subplots()
    ax.scatter(ym_te, best_pred, s=6, alpha=0.3, color="#1f77b4")
    lim = [min(ym_te.min(), best_pred.min()), max(ym_te.max(), best_pred.max())]
    ax.plot(lim, lim, "r--", lw=1)
    ax.set_xlabel("真实 trq_margin")
    ax.set_ylabel("预测 trq_margin")
    ax.set_title(f"预测 vs 真实（{best_name}, R²={best_r2:.3f}）")
    savefig("pred_vs_true", "scenario_engine")

    # ---- 分类：故障检测 ----
    print("\n=== 分类任务：faulty（故障检测）===")
    clf_results = {}
    for name, model in classifiers.items():
        model.fit(X_tr, yf_tr)
        proba = model.predict_proba(X_te)[:, 1]
        pred = (proba >= 0.5).astype(int)
        auc = float(roc_auc_score(yf_te, proba))
        acc = float(accuracy_score(yf_te, pred))
        clf_results[name] = (auc, acc)
        print(f"  {name:14s} ROC-AUC={auc:.4f}  准确率={acc:.4f}")

    # ---- 特征重要性（RF 回归）----
    rf = regressors["RandomForest"]
    importances = rf.feature_importances_
    order = np.argsort(importances)[::-1]
    fig, ax = plt.subplots()
    ax.barh([FEATURES[i] for i in order][::-1], importances[order][::-1], color="#ff7f0e")
    ax.set_xlabel("重要性")
    ax.set_title("RandomForest 特征重要性（扭矩裕度）")
    savefig("feature_importance", "scenario_engine")

    print(f"\n[总结] 回归最佳：{best_name} (R²={best_r2:.4f})；"
          f"故障检测 ROC-AUC 最高 {max(v[0] for v in clf_results.values()):.4f}")
    print(f"[总结] 关键健康相关特征：{FEATURES[order[0]]}, {FEATURES[order[1]]}")


if __name__ == "__main__":
    main()
