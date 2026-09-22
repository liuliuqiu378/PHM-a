"""
模块 06：工程实战（三）—— 概念漂移 + 主动学习闭环（模型训练 / 数据闭环）
==========================================================================================

你原话：「模型训练（持续学习/数据闭环）」是最大缺口。本脚本把这一闭环做成**可运行**的 demo，
回答两件事：

  1) 概念漂移（Concept Drift）：模型上线后，工况变了（负载/温度/原料/机型），
     原本学到的「健康 vs 故障」边界悄悄旋转，模型在「自己不知情」下持续退化。
  2) 主动学习（Active Learning）闭环：模型发现自己「拿不准」(预测置信度掉到边界附近)，
     就挑「信息量最大」的样本请专家打标，增量重训——而不是把所有样本盲目全标。

映射 PHM：特征 = (振动RMS, 温度)，真实判别边界随工况旋转（同一 RMS 在不同温度下
故障阈值不同）。demo 用 2D 二分类直观可视化；真实可用同框架套到 RUL/分类网络。

三种策略对比（同一数据流）：
  · no_retrain   ：上线后永不更新 → 漂移后误差持续攀升（灾难）。
  · random       ：每轮随机挑 K 个样本打标重训 → 能恢复，但浪费标签在冗余样本。
  · active       ：每轮挑「最不确定」(置信度最接近 0.5) 的样本打标重训 → 同样标签数恢复更快。

运行（从 PHM 根目录）：
  python src/module_06_engineering/drift_active_learning.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

from src.common.plot_style import set_style, savefig

N_CHECK = 16        # 部署时间分 16 个检查点（"周"）
PER = 40            # 每检查点到达 40 个未标注样本
K = 4               # 每轮主动/随机采样打标数（标签稀缺，才显出「挑哪条」的价值）
RETAIN = 48         # 重训只用最近 RETAIN 条标签（滑动窗口，模拟在线增量、忘掉旧工况）
DRIFT_DEG = 38.0    # 整个部署期真实边界旋转总角度（工况漂移幅度）
SEED = 0


# ------------------------- 1. 数据生成（真边界随时间旋转） -------------------------
def true_boundary(t):
    """检查点 t 的真实判别边界：权重向量从 0° 线性旋转到 DRIFT_DEG。"""
    ang = np.deg2rad(DRIFT_DEG * t / (N_CHECK - 1))
    w = np.array([np.cos(ang), np.sin(ang)])      # 单位法向，旋转
    return w, 0.0


def sample_batch(t, n, rng):
    """在 2D 特征空间均匀采样 n 个样本，按当前真边界给标签。"""
    X = rng.normal(0, 1.4, size=(n, 2))
    w, b = true_boundary(t)
    y = (X @ w + b > 0).astype(int)
    return X, y, w


def uncertainty(model, X):
    """模型预测置信度离 0.5 的距离：越小越「拿不准」（主动学习要挑这种）。"""
    p = model.predict_proba(X)[:, 1]
    return np.abs(p - 0.5)


# ------------------------- 2. 三种策略模拟 -------------------------
def run(seed=SEED):
    rng = np.random.default_rng(seed)
    # 初始训练集（上线前在 t=0 工况下标好）
    X0, y0, _ = sample_batch(0, 120, rng)
    w0, _ = true_boundary(0)

    results = {}
    for strategy in ("no_retrain", "random", "active"):
        rng2 = np.random.default_rng(seed + hash(strategy) % 1000)
        if strategy == "no_retrain":
            model = LogisticRegression(max_iter=300).fit(X0, y0)
        else:
            model = LogisticRegression(max_iter=300).fit(X0, y0)
        # 维护各自的「已标注池」
        Xlab, ylab = X0.copy(), y0.copy()
        err_curve, unc_curve, nlab_curve = [], [], []
        queried_pts = []          # 记录主动策略挑中的样本（画图用）
        rolling_unc = []

        for t in range(N_CHECK):
            Xb, yb, w_t = sample_batch(t, PER, rng2)
            # 当前误差（oracle 知道真标签，仅用于评估，不泄露给模型）
            err = 1 - model.score(Xb, yb)
            err_curve.append(err)
            unc = uncertainty(model, Xb).mean()
            unc_curve.append(unc)
            rolling_unc.append(unc)

            if strategy != "no_retrain" and t >= 3:
                # 漂移触发：滚动不确定度超过初始基线 → 进入主动采样
                if np.mean(rolling_unc[-3:]) > 0.16:
                    if strategy == "active":
                        order = np.argsort(uncertainty(model, Xb))      # 最不确定优先
                    else:
                        order = rng2.permutation(PER)
                    pick = order[:K]
                    Xlab = np.vstack([Xlab, Xb[pick]])
                    ylab = np.concatenate([ylab, yb[pick]])
                    if strategy == "active":
                        queried_pts.extend(Xb[pick].tolist())
                    # 滑动窗口重训：只用最近 RETAIN 条（忘掉已漂移的旧工况）
                    if len(Xlab) > RETAIN:
                        model = LogisticRegression(max_iter=300).fit(Xlab[-RETAIN:], ylab[-RETAIN:])
                    else:
                        model = LogisticRegression(max_iter=300).fit(Xlab, ylab)
            nlab_curve.append(len(Xlab))

        results[strategy] = dict(err=err_curve, unc=unc_curve,
                                 nlab=nlab_curve, model=model,
                                 w0=w0, queried=np.array(queried_pts))
    return results, X0, y0


# ------------------------- 3. 图 -------------------------
def main():
    set_style()
    results, X0, y0 = run()

    fig, ax = plt.subplots(1, 2, figsize=(14, 5.5))

    # 左：误差随时间（漂移 onset + 三种策略恢复对比）
    x = np.arange(N_CHECK)
    ax[0].plot(x, results["no_retrain"]["err"], "k--", lw=2, label="永不重训（灾难）")
    ax[0].plot(x, results["random"]["err"], color="#ff7f0e", lw=2, label="随机采样重训")
    ax[0].plot(x, results["active"]["err"], color="#2ca02c", lw=2.2, label="主动学习（挑最不确定）")
    ax[0].axvspan(3, N_CHECK - 1, color="#d62728", alpha=0.07, label="工况漂移区")
    ax[0].set_xlabel("部署检查点（时间 →）")
    ax[0].set_ylabel("测试误差")
    ax[0].set_title("概念漂移下：不更新会崩，主动学习最快恢复")
    ax[0].legend(fontsize=8)

    # 右：决策边界（初始 vs 主动学习最终）+ 主动挑中的样本
    res_a = results["active"]
    w0 = res_a["w0"]
    # 画初始边界（上线前）
    xx = np.linspace(-3, 3, 50)
    ax[1].plot(xx, -w0[0] / (w0[1] + 1e-9) * xx, "b--", lw=1.5, label="初始边界(t=0)")
    # 画主动学习最终边界
    m = res_a["model"]
    wf, bf = m.coef_[0], m.intercept_[0]
    ax[1].plot(xx, -(wf[0] / (wf[1] + 1e-9)) * xx - bf / (wf[1] + 1e-9),
               "g-", lw=2, label="主动学习后边界")
    # 初始训练点
    ax[1].scatter(X0[:, 0], X0[:, 1], c=y0, cmap="coolwarm", s=12, alpha=0.4,
                  edgecolors="none", label="初始标注样本")
    # 主动挑中的样本（信息量最大）
    q = res_a["queried"]
    if len(q):
        ax[1].scatter(q[:, 0], q[:, 1], facecolors="none", edgecolors="k",
                      s=70, linewidths=1.4, label="主动挑中(请专家标)")
    ax[1].set_xlabel("特征1（如 振动RMS）")
    ax[1].set_ylabel("特征2（如 温度）")
    ax[1].set_title("主动学习：只把「边界附近最不确定」的样本送专家")
    ax[1].legend(fontsize=8, loc="upper right")

    savefig("drift_active_learning", "engineering")

    # 打印关键数字
    print("[漂移] 真实边界在部署期旋转了 %.0f°（工况变化）" % DRIFT_DEG)
    print(f"[no_retrain] 末点误差 = {results['no_retrain']['err'][-1]:.3f}（持续恶化）")
    print(f"[random]     末点误差 = {results['random']['err'][-1]:.3f}  累计标签 = {results['random']['nlab'][-1]}")
    print(f"[active]     末点误差 = {results['active']['err'][-1]:.3f}  累计标签 = {results['active']['nlab'][-1]}（同量标签，恢复更快）")
    print("\n[结论] 上线≠结束：工况漂移让模型静默退化；主动学习只把「最不确定」的边界样本")
    print("        送专家打标、增量重训，用同样甚至更少的标签把误差拉回——这就是「数据闭环」。")


if __name__ == "__main__":
    main()
