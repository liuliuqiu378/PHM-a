"""
迁移学习/域适应直观 demo：为什么"一台机器训练的模型到另一台/另一种工况就掉点"，以及怎么救。

核心演示：
  故障调制幅度随负载(工况)成比例放大 → 绝对幅值特征(RMS/峰值)发生"分布偏移(covariate shift)"
  → 在源域训的阈值到目标域失效（掉点）；
  改用"负载无关"的归一化特征(峰值/RMS)后，分布对齐、掉点被救回。

这与教程 [`03_诊断`](../docs/03_diagnosis.md) 的跨工况暴跌( RF 0.977→0.673 )是同一现象，
且点出 PHM 独有优势：BPFI 等故障*频率*是负载无关的，只有*幅值*随负载变。

运行：python src/module_03_diagnosis/demo_domain_shift.py
依赖：numpy, scipy, sklearn, matplotlib
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.common.plot_style import set_style, savefig


def make_domain(rng, n_each, load, n=240, impulse=8.0):
    """造一个域：load 决定整体幅度（模拟不同负载/不同机器的传递增益）。

    健康 = 高斯噪声；故障 = 高斯噪声 + 一个瞬态冲击(模拟轴承点蚀的一次敲击)。
    整段信号乘 load → 绝对幅值(RMS)随 load 漂移；
    但'峰值/RMS'对冲击类故障是*负载无关*的（分子分母同乘 load 抵消）。
    """
    X_abs, X_norm, y = [], [], []
    for _ in range(n_each):
        for cls in (0, 1):  # 0=健康, 1=故障
            sig = rng.normal(0, 1.0, n)
            if cls == 1:
                sig[rng.integers(0, n)] += impulse   # 瞬态冲击
            sig = load * sig
            rms = np.sqrt(np.mean(sig**2))
            peak = np.max(np.abs(sig))
            X_abs.append(rms)                       # 绝对幅值特征（随 load 漂移）
            X_norm.append(peak / (rms + 1e-9))      # 归一化特征（负载无关）
            y.append(cls)
    return np.array(X_abs).reshape(-1, 1), np.array(X_norm).reshape(-1, 1), np.array(y)


def evaluate():
    set_style()
    rng = np.random.default_rng(0)
    # 源域（实验室/机器A，load=1.0）：造足量有标签数据
    Xa_abs, Xa_norm, ya = make_domain(rng, 400, load=1.0)
    # 目标域（现场/机器B，load=2.2）：分布偏移（幅度整体变大）
    Xb_abs, Xb_norm, yb = make_domain(rng, 400, load=2.2)

    acc_src, acc_tgt = {}, {}
    for name, Xa, Xb in [("绝对幅值特征(RMS)", Xa_abs, Xb_abs),
                         ("归一化特征(峰值/RMS)", Xa_norm, Xb_norm)]:
        clf = RandomForestClassifier(n_estimators=120, random_state=0)
        clf.fit(Xa, ya)
        acc_src[name] = accuracy_score(ya, clf.predict(Xa))
        acc_tgt[name] = accuracy_score(yb, clf.predict(Xb))

    # 图：两个域的特征分布对比（绝对 vs 归一化）
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, Xa, Xb, title in [
        (axes[0], Xa_abs, Xb_abs, "绝对幅值特征：源/目标域明显错开(分布偏移)"),
        (axes[1], Xa_norm, Xb_norm, "归一化特征：源/目标域基本重叠(对齐)"),
    ]:
        ax.hist(Xa[yb == 0], bins=30, alpha=0.5, label="源域-健康", color="#27ae60")
        ax.hist(Xa[yb == 1], bins=30, alpha=0.5, label="源域-故障", color="#c0392b")
        ax.hist(Xb[yb == 0], bins=30, alpha=0.4, histtype="step", lw=2, label="目标域-健康", color="#27ae60")
        ax.hist(Xb[yb == 1], bins=30, alpha=0.4, histtype="step", lw=2, label="目标域-故障", color="#c0392b")
        ax.set_title(title)
        ax.set_xlabel("特征值")
        ax.set_ylabel("样本数")
        ax.legend(fontsize=8)
    plt.tight_layout()
    savefig("domain_shift_demo", subdir="diagnosis")
    plt.close()

    print("=== 准确率（源域=机器A训练，目标域=机器B测试）===")
    for k in acc_src:
        print(f"  {k:24s}: 源域内 {acc_src[k]*100:.1f}%  |  跨域 {acc_tgt[k]*100:.1f}%")
    print("结论：绝对幅值特征因负载漂移而跨域暴跌；归一化(负载无关)特征对齐后救回。")
    return acc_tgt


if __name__ == "__main__":
    evaluate()
