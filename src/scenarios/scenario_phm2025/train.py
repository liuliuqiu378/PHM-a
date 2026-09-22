"""
场景 E（PHM2025 北美挑战赛）：涡扇发动机剩余循环预测（真·RUL）
============================================================================

数据：`data/PHM2025/training_data.csv`（已随挑战赛包提供，5.97 万行）。
每行是一台发动机（ESN）在某次飞行循环（Snapshot）的「多变量传感器快照」，
目标是预测到 3 个维护事件的剩余循环数：
  - Cycles_to_WW     到 Windmill / 某事件
  - Cycles_to_HPC_SV 到高压压气机突防（单喘振？）事件
  - Cycles_to_HPT_SV 到高压涡轮突防事件

与场景 B（PHM2024，近乎确定性、AUC=1.0）不同，这里是**连续 RUL 回归**，
且只有 4 台发动机（真实数据稀缺），所以必须做 **按发动机留一法（LOO）交叉验证**，
否则同一台发动机同时出现在训练/测试集会严重高估精度。

本脚本演示端到端真实 RUL 流程：
  1) 加载 + 缺失值处理（按训练折均值填充，防泄漏）；
  2) 可视化一台发动机的退化趋势；
  3) 按 ESN 留一法 CV，训练 RandomForest 回归 3 个 RUL 目标；
  4) 报告各目标 RMSE、预测-真实散点、特征重要性。

运行（从 PHM 根目录）：
  python src/scenarios/scenario_phm2025/train.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error, r2_score

from src.common.plot_style import set_style, savefig

CSV = ROOT / "data" / "PHM2025" / "training_data.csv"
TARGETS = ["Cycles_to_WW", "Cycles_to_HPC_SV", "Cycles_to_HPT_SV"]
SENSED = ["Sensed_Altitude", "Sensed_Mach", "Sensed_Pamb", "Sensed_Pt2",
          "Sensed_TAT", "Sensed_WFuel", "Sensed_VAFN", "Sensed_VBV",
          "Sensed_Fan_Speed", "Sensed_Core_Speed", "Sensed_T25", "Sensed_T3",
          "Sensed_Ps3", "Sensed_T45", "Sensed_P25", "Sensed_T5"]
EXTRA = ["Cycles_Since_New", "Cumulative_WWs", "Cumulative_HPC_SVs",
         "Cumulative_HPT_SVs"]
FEATURES = SENSED + EXTRA


def load():
    df = pd.read_csv(CSV)
    df = df.dropna(subset=TARGETS)  # 目标无缺失（EDA 已确认）
    df[FEATURES] = df[FEATURES].astype(float)
    return df


def main():
    set_style()
    df = load()
    esn = df["ESN"].values
    X = df[FEATURES].values.astype(np.float32)
    y = df[TARGETS].values.astype(np.float32)
    print(f"[data] 样本={len(df)}  发动机={np.unique(esn).__len__()}  "
          f"特征={len(FEATURES)}  目标={TARGETS}")

    # ---- 1) 退化趋势（第一台发动机）----
    e0 = np.unique(esn)[0]
    sub = df[df.ESN == e0].sort_values("Snapshot")
    fig, ax = plt.subplots(figsize=(9, 4))
    for t, c in zip(TARGETS, ["#1f77b4", "#ff7f0e", "#2ca02c"]):
        ax.plot(sub["Snapshot"], sub[t], label=t, color=c, lw=1.3)
    ax.set_xlabel("Snapshot（飞行循环序号）")
    ax.set_ylabel("剩余循环数")
    ax.set_title(f"发动机 {e0} 的剩余循环退化趋势（RUL）")
    ax.legend()
    savefig("degradation_trend", "phm2025")

    # ---- 2) 按 ESN 留一法 CV，回归 3 个 RUL ----
    gkf = GroupKFold(n_splits=len(np.unique(esn)))  # 留一发动机
    rng = np.random.RandomState(42)
    rmse_all = {t: [] for t in TARGETS}
    r2_all = {t: [] for t in TARGETS}
    pooled = {t: ([], []) for t in TARGETS}  # 拼接各折测试集的预测/真实

    for t in TARGETS:
        for tr, te in gkf.split(X, y[:, TARGETS.index(t)], groups=esn):
            mu = np.nanmean(X[tr], axis=0)
            Xtr = np.where(np.isnan(X[tr]), mu, X[tr])
            Xte = np.where(np.isnan(X[te]), mu, X[te])
            m = RandomForestRegressor(n_estimators=200, n_jobs=-1,
                                      random_state=0, min_samples_leaf=3)
            m.fit(Xtr, y[tr, TARGETS.index(t)])
            pred = m.predict(Xte)
            rmse_all[t].append(float(np.sqrt(mean_squared_error(y[te, TARGETS.index(t)], pred))))
            r2_all[t].append(float(r2_score(y[te, TARGETS.index(t)], pred)))
            pooled[t][0].append(y[te, TARGETS.index(t)])
            pooled[t][1].append(pred)
        print(f"[cv]   {t:18s}  RMSE(mean±std)={np.mean(rmse_all[t]):7.1f} ± "
              f"{np.std(rmse_all[t]):5.1f}  R²(mean)={np.mean(r2_all[t]):+.3f}")

    # ---- 3) 预测-真实散点（最大范围目标 HPC_SV）----
    tgt = "Cycles_to_HPC_SV"
    yt = np.concatenate(pooled[tgt][0]); pt = np.concatenate(pooled[tgt][1])
    fig, ax = plt.subplots(figsize=(5.2, 5))
    ax.scatter(yt, pt, s=6, alpha=0.3, color="#1f77b4")
    lim = [min(yt.min(), pt.min()), max(yt.max(), pt.max())]
    ax.plot(lim, lim, "r--", lw=1)
    ax.set_xlabel("真实剩余循环")
    ax.set_ylabel("预测剩余循环")
    ax.set_title(f"{tgt}：留一法预测 vs 真实")
    savefig("rul_scatter", "phm2025")

    # ---- 4) 特征重要性（用全部数据训练，仅展示排序）----
    mu = np.nanmean(X, axis=0)
    Xf = np.where(np.isnan(X), mu, X)
    imp_m = RandomForestRegressor(200, n_jobs=-1, random_state=0,
                                  min_samples_leaf=3).fit(Xf, y[:, TARGETS.index(tgt)])
    order = np.argsort(imp_m.feature_importances_)[::-1][:12]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh([FEATURES[i] for i in order][::-1],
            imp_m.feature_importances_[order][::-1], color="#2ca02c")
    ax.set_title(f"{tgt} 特征重要性（Top-12）")
    ax.set_xlabel("importance")
    savefig("feature_importance", "phm2025")

    # ---- 5) CV RMSE 汇总图（各目标均值±std）----
    fig, ax = plt.subplots(figsize=(7, 4))
    means = [np.mean(rmse_all[t]) for t in TARGETS]
    stds = [np.std(rmse_all[t]) for t in TARGETS]
    x = np.arange(len(TARGETS))
    ax.bar(x, means, yerr=stds, capsize=5,
           color=["#1f77b4", "#ff7f0e", "#2ca02c"])
    for i, m in enumerate(means):
        ax.text(i, m + stds[i] + 20, f"{m:.0f}", ha="center")
    ax.set_xticks(x); ax.set_xticklabels(TARGETS, rotation=15)
    ax.set_ylabel("RMSE（剩余循环）")
    ax.set_title("各目标留一法 RMSE（按发动机交叉验证，均值±std，4 折）")
    savefig("cv_rmse", "phm2025")

    print(f"\n[结论] PHM2025 是真实 RUL 回归：4 台发动机、按发动机留一法 CV，"
          f"RMSE 在百~千循环量级（目标本身跨度数百~数千），明显比场景 B 的"
          f"「近确定性」更难、更接近现场真实水平。")


if __name__ == "__main__":
    main()
