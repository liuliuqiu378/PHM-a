"""信号处理进阶三件套：阶次跟踪 / 谱峭度(Kurtogram) / 调制与边带。

对应文档 docs/12_signal_advanced.md。每个 demo 都用 numpy+scipy 合成信号，
把"为什么需要这种方法"可视化出来——重点不是调包，是建立物理直觉。

运行：python src/module_02_signal/demo_advanced_signal.py
生成：figures/signal/advanced_order.png
      figures/signal/advanced_kurtogram.png
      figures/signal/advanced_modulation.png
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert, butter, filtfilt, stft
from scipy.stats import kurtosis

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.common.plot_style import set_style, savefig


# ----------------------------------------------------------------------------
# 1) 阶次跟踪 (Order Tracking)：变速下，FFT 会糊，按"转角"重采样才锁得住
# ----------------------------------------------------------------------------
def demo_order_tracking():
    fs = 12000.0
    dur = 2.0
    N = int(fs * dur)
    t = np.arange(N) / fs

    # 转速从 1200 线性升到 1800 rpm（真实机器启停/变负载常态）
    rpm = 1200 + 600 * np.linspace(0, 1, N)
    fr = rpm / 60.0                 # 转频 (rev/s)
    ang = np.cumsum(fr) / fs        # 累计转角 (圈)，对 fr 积分

    # 一个"每转发生 3 次"的故障（3 阶）→ 频率 = 3*fr(t)，随转速扫动 60→90Hz
    x_fault = np.sin(2 * np.pi * 3.0 * ang)
    # 一个固定频率成分（如电气 2400Hz 哼声），与转速无关
    x_hum = 0.4 * np.sin(2 * np.pi * 2400.0 * t)
    x = x_fault + x_hum + 0.05 * np.random.randn(N)

    # 常规 FFT（Hz 域）：故障被转速扫宽，糊成一片
    X = np.fft.rfft(x)
    freq = np.fft.rfftfreq(N, d=1 / fs)

    # 阶次谱：按"均匀转角"重采样到角域，再做 FFT
    Nrev = ang[-1] - ang[0]
    ang_uni = np.linspace(ang[0], ang[-1], N)
    x_uni = np.interp(ang_uni, ang, x)
    Xord = np.fft.rfft(x_uni)
    orders = np.fft.rfftfreq(N, d=1.0 / Nrev)   # 单位：阶 (次/转)

    set_style()
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 7))
    a1.plot(freq[: N // 8], np.abs(X[: N // 8]))
    a1.set_title("常规 FFT（Hz 域）：故障频率随转速扫动 → 糊成宽带")
    a1.set_xlabel("频率 / Hz"); a1.set_ylabel("幅值")
    a1.axvline(2400, color="C2", ls="--", lw=1, label="2400Hz 固定哼声(清晰)")
    a1.axvspan(60, 90, color="C1", alpha=0.2, label="故障 3×fr 被扫宽到 60~90Hz")
    a1.legend(fontsize=8)

    a2.plot(orders[: N // 6], np.abs(Xord[: N // 6]))
    a2.set_title("阶次谱（角域）：故障锁定在 3 阶；固定哼声反而被糊 (不随转角同步)")
    a2.set_xlabel("阶次 (次/转)"); a2.set_ylabel("幅值")
    a2.axvline(3, color="C1", ls="--", lw=1.5, label="故障 = 3 阶（清晰单线）")
    a2.legend(fontsize=8)

    savefig("advanced_order", "signal")
    plt.close()


# ----------------------------------------------------------------------------
# 2) 谱峭度 / Kurtogram：自动找"故障最响"的共振带，再包络解调
# ----------------------------------------------------------------------------
def demo_spectral_kurtosis():
    fs = 12000.0
    dur = 1.0
    N = int(fs * dur)
    t = np.arange(N) / fs

    # 故障：周期性冲击，每次激发 3500Hz 附近的结构共振（阻尼正弦）
    fault_f = 120.0
    fc_res = 3500.0
    imp = np.zeros(N)
    for kk in np.arange(0, dur, 1 / fault_f):
        imp[int(kk * fs)] = 1.0
    tau = np.arange(0, 0.02, 1 / fs)
    ring = np.exp(-800 * tau) * np.sin(2 * np.pi * fc_res * tau)
    sig = np.convolve(imp, ring, mode="same")[:N]
    sig = sig + 0.9 * np.random.randn(N)        # 强背景噪声淹没冲击
    sig = sig / np.std(sig)

    # 谱峭度：STFT 幅值在时间维的峭度，哪里峭度高=瞬态能量集中=共振带
    f, tt, Z = stft(sig, fs=fs, nperseg=512, noverlap=384)
    env = np.abs(Z)
    SK = kurtosis(env, axis=1)                   # 每频率一条 SK 曲线
    best_f = f[np.nanargmax(SK)]

    # 选带通：最优共振带 vs 一个"错"的低频带
    def band_envelope(lo, hi):
        b, a = butter(4, [lo / (fs / 2), hi / (fs / 2)], btype="band")
        y = filtfilt(b, a, sig)
        return np.abs(hilbert(y))

    env_good = band_envelope(max(200, best_f - 600), best_f + 600)
    env_bad = band_envelope(400, 1000)
    ef = np.fft.rfftfreq(N, d=1 / fs)
    Eg = np.abs(np.fft.rfft(env_good))
    Eb = np.abs(np.fft.rfft(env_bad))

    set_style()
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    ax[0, 0].plot(f, SK)
    ax[0, 0].axvline(best_f, color="C1", ls="--", label=f"SK 最大 ≈ {best_f:.0f}Hz")
    ax[0, 0].set_title("谱峭度 SK(f)：自动定位共振带"); ax[0, 0].set_xlabel("Hz"); ax[0, 0].legend(fontsize=8)
    ax[0, 1].plot(ef[: 800], Eg[: 800], label=f"最优带({best_f:.0f}Hz)")
    ax[0, 1].plot(ef[: 800], Eb[: 800], color="C3", label="错带(400~1000Hz)")
    ax[0, 1].axvline(fault_f, color="k", ls=":", label=f"故障频率 {fault_f:.0f}Hz")
    ax[0, 1].set_title("包络谱对比：最优带清晰出故障线，错带被噪声淹没")
    ax[0, 1].set_xlabel("Hz"); ax[0, 1].legend(fontsize=8)
    # 时域展示冲击被噪声淹没
    ax[1, 0].plot(t[: 2000], sig[: 2000])
    ax[1, 0].set_title("时域：冲击被强噪声淹没，肉眼看不见"); ax[1, 0].set_xlabel("s")
    ax[1, 1].plot(t[: 2000], env_good[: 2000])
    ax[1, 1].set_title("最优带包络：周期性冲击被提取出来"); ax[1, 1].set_xlabel("s")
    fig.suptitle("谱峭度 = 把 '选对带通' 从靠人试变成自动定位", fontsize=13)
    savefig("advanced_kurtogram", "signal")
    plt.close()


# ----------------------------------------------------------------------------
# 3) 调制与边带：AM/FM 怎么长出边带，内圈 vs 外圈因此不同
# ----------------------------------------------------------------------------
def demo_modulation():
    fs = 12000.0
    dur = 0.1
    N = int(fs * dur)
    t = np.arange(N) / fs
    fc = 3000.0          # 载波（如啮合/共振频率）
    ff = 50.0            # 调制频率（如转频 fr）

    # 幅度调制 AM：包络随调制频率起伏 → 频谱出现 fc ± ff 两根边带
    x_am = (1 + 0.8 * np.cos(2 * np.pi * ff * t)) * np.cos(2 * np.pi * fc * t)
    # 频率调制 FM：瞬时频率被调制 → 频谱出现 fc ± n·ff 多根边带
    beta = 3.0
    x_fm = np.cos(2 * np.pi * fc * t + beta * np.sin(2 * np.pi * ff * t))

    def show(x, ax, title):
        X = np.abs(np.fft.rfft(x))
        fr = np.fft.rfftfreq(N, d=1 / fs)
        m = (fr > fc - 400) & (fr < fc + 400)
        ax.plot(fr[m], X[m])
        ax.set_title(title); ax.set_xlabel("Hz"); ax.set_ylabel("幅值")

    set_style()
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 7))
    show(x_am, a1, "AM：载波 fc 两侧长出 ff 间距的边带（内圈故障随转频旋转→被 fr 调制）")
    a1.axvline(fc, color="k", ls=":", lw=1); a1.axvline(fc - ff, color="C1", ls="--", lw=1); a1.axvline(fc + ff, color="C1", ls="--", lw=1)
    a1.annotate("fc-ff", (fc - ff, 0), (fc - ff, a1.get_ylim()[1]*0.7), fontsize=8, color="C1")
    show(x_fm, a2, "FM：载波两侧长出 ff 整数倍的多阶边带（齿轮啮合被转频调制）")
    for n in range(1, 5):
        a2.axvline(fc - n * ff, color="C1", ls=":", lw=0.8); a2.axvline(fc + n * ff, color="C1", ls=":", lw=0.8)
    a2.axvline(fc, color="k", ls=":", lw=1)
    savefig("advanced_modulation", "signal")
    plt.close()


if __name__ == "__main__":
    demo_order_tracking()
    demo_spectral_kurtosis()
    demo_modulation()
    print("done: 3 advanced-signal figures in figures/signal/")
