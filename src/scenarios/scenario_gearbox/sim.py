"""
场景 D：齿轮箱健康评估（振动信号物理仿真 + 故障诊断）
=========================================================================

没有本地齿轮箱数据，这里用**物理可解释的振动仿真**生成数据，好处是开箱即跑、
机理透明，且能精确对照「齿轮啮合频率 / 边带」这些课本概念：

  - 齿轮啮合频率 GMF = 齿数 Z × 轴转频 f_shaft（每对齿啮合一次产生一个冲击）
  - 健康齿轮：振动是 GMF 各次谐波 + 高斯噪声，波形「平稳」。
  - 故障齿轮（断齿/局部剥落）：每转一圈，故障齿参与啮合时多一次强冲击，
    相当于对 GMF 做 f_shaft 的幅值调制 → 在 GMF 两侧出现 f_shaft 间隔的「边带」，
    且时域峭度(kurtosis)显著升高（冲击让尖峰变多）。

诊断思路（与模块03 一脉相承）：
  时域特征(峭度/RMS) + 频域(边带能量占比) → 随机森林分类 健康/故障。
并画 FFT 与包络谱，肉眼可见边带来验证「物理是对的」。

运行：从 PHM 根目录执行
  python src/scenarios/scenario_gearbox/sim.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib.pyplot as plt

from src.common.plot_style import set_style, savefig
from src.common.signal_utils import fft_spectrum, envelope_spectrum
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix


def gen_signal(rpm, z_teeth, faulty, fs=12000.0, dur=1.0, seed=0):
    """生成一段齿轮箱振动信号。"""
    rng = np.random.default_rng(seed)
    t = np.arange(int(fs * dur)) / fs
    f_shaft = rpm / 60.0
    gmf = z_teeth * f_shaft

    # 健康基分量：GMF 的若干谐波
    x = np.zeros_like(t)
    for h in range(1, 6):
        x += (1.0 / h) * np.sin(2 * np.pi * h * gmf * t + rng.uniform(0, 2 * np.pi))
    x *= 0.5

    if faulty:
        # 故障：每转一次额外冲击（f_shaft 调制），形成边带、抬升峭度。
        # 用「衰减振荡核」与冲击序列卷积，模拟冲击经齿轮/轴承传递后的局部铃振。
        period = 1.0 / f_shaft
        idx = np.arange(0, len(t), max(1, int(period * fs)))
        impulse_train = np.zeros_like(t)
        impulse_train[idx] = 1.0
        M = max(20, int(0.005 * fs))          # 铃振持续 ~5ms
        tau = M / 6.0
        kern = np.exp(-np.arange(M) / tau) * np.sin(2 * np.pi * gmf * np.arange(M) / fs)
        fault_comp = 1.6 * np.convolve(impulse_train, kern, mode="same")
        x += fault_comp

    x += rng.normal(0, 0.05, size=t.shape)  # 背景噪声
    return x, fs, gmf, f_shaft


def features(x, fs, gmf, f_shaft):
    mean = x.mean()
    std = x.std()
    kurt = np.mean((x - mean) ** 4) / (std ** 4 + 1e-12)
    rms = np.sqrt(np.mean(x ** 2))
    freq, mag = fft_spectrum(x, fs)
    # 边带能量占比：GMF ± 3*f_shaft 内相对 GMF 主峰
    main = mag[np.argmin(np.abs(freq - gmf))]
    side = 0.0
    for k in range(1, 4):
        for sgn in (1, -1):
            f = gmf + sgn * k * f_shaft
            side += mag[np.argmin(np.abs(freq - f))]
    return [float(kurt), float(rms), float(side / (main + 1e-12))]


def main():
    set_style()
    rpm, z = 1800, 24
    # 生成健康/故障样本各若干
    X, y = [], []
    for label, faulty in [(0, False), (1, True)]:
        for i in range(60):
            x, fs, gmf, fsh = gen_signal(rpm, z, faulty, seed=1000 + label * 100 + i)
            X.append(features(x, fs, gmf, fsh))
            y.append(label)
    X = np.array(X)
    y = np.array(y)
    print(f"[data] 样本 {X.shape}, 健康/故障各 {int((y == 0).sum())}/{int((y == 1).sum())}")

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
    clf = RandomForestClassifier(n_estimators=200, random_state=42).fit(Xtr, ytr)
    acc = accuracy_score(yte, clf.predict(Xte))
    print(f"[诊断] 随机森林准确率 {acc:.4f}")
    print("  特征重要性:", dict(zip(["峭度", "RMS", "边带占比"], clf.feature_importances_.round(3))))

    # ---- 图1：健康 vs 故障 时域 ----
    xh, fsh_, _, _ = gen_signal(rpm, z, False, seed=7)
    xf, fsf, _, _ = gen_signal(rpm, z, True, seed=7)
    fig, ax = plt.subplots(2, 1, sharex=False, figsize=(10, 5))
    ax[0].plot(xh[:1000], color="#2ca02c")
    ax[0].set_title(f"健康齿轮 时域（峭度={np.mean((xh-xh.mean())**4)/(xh.std()**4):.2f}）")
    ax[1].plot(xf[:1000], color="#d62728")
    ax[1].set_title(f"故障齿轮 时域（峭度={np.mean((xf-xf.mean())**4)/(xf.std()**4):.2f}，冲击明显）")
    savefig("gear_time", "scenario_gearbox")

    # ---- 图2：包络谱对比（看边带）----
    gmf = z * (rpm / 60.0)
    fsh = rpm / 60.0
    eh, emh = envelope_spectrum(xh, fsh_)
    ef, emf = envelope_spectrum(xf, fsf)
    fig, ax = plt.subplots(figsize=(10, 4))
    band = (ef >= gmf - 4 * fsh) & (ef <= gmf + 4 * fsh)
    ax.plot(ef[band], emh[band], color="#2ca02c", label="健康")
    ax.plot(ef[band], emf[band], color="#d62728", label="故障")
    for k in range(-3, 4):
        ax.axvline(gmf + k * fsh, color="gray", ls=":", lw=0.8)
    ax.set_xlabel("频率 (Hz)")
    ax.set_ylabel("包络幅值")
    ax.set_title(f"包络谱：GMF={gmf:.0f}Hz 两侧边带（故障侧出现 f_shaft={fsh:.0f}Hz 间隔）")
    ax.legend()
    savefig("gear_envelope", "scenario_gearbox")


if __name__ == "__main__":
    main()
