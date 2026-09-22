"""
P1/P2 进阶信号专题可跑 demo（合成信号，仅用于直观演示）：
  1) 同步平均 (Time Synchronous Averaging, TSA)
  2) 倒频谱 (Cepstrum) —— 把"一堆等间距边带"压成一个峰
  3) 转子动力学频谱指纹（1x 不平衡 / 2x 不对中 / 亚同步 油膜涡动）

运行：python src/module_02_signal/demo_advanced2_signal.py
依赖：numpy, scipy, matplotlib（与教程一致）
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.common.plot_style import set_style, savefig


def demo_tsa():
    """同步平均：按'每转'对齐、求平均，把与转速不同步的噪声抹掉。"""
    set_style()
    rng = np.random.default_rng(0)
    fs = 12000
    n_cycles = 40
    per = 500                      # 每转采样点数（即转频 = fs/per = 24 Hz）
    n = n_cycles * per
    t = np.arange(n) / fs

    # 每转一次的齿轮啮合 + 一处局部缺陷（每转一个尖脉冲）
    clean = 0.6 * np.sin(2 * np.pi * (fs / per) * t)          # 啮合基频
    defect = np.zeros(n)
    for c in range(n_cycles):
        defect[c * per + 120] = 1.0                           # 每转固定在相位 120 处冲击
    clean += 0.5 * defect
    noise = rng.normal(0, 0.8, n)                            # 强背景噪声（含别台机器）
    raw = clean + noise

    # 同步平均：切成 n_cycles 段，每段 per 点，逐点平均
    blocks = raw.reshape(n_cycles, per)
    avg = blocks.mean(axis=0)

    tt = (np.arange(per)) / fs * 1e3
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    axes[0].plot(tt, raw[:per], color="#888", lw=0.6, label="单转原始（强噪声淹没缺陷）")
    axes[0].set_title("原始单转信号（噪声把每转的缺陷冲击盖住了）")
    axes[0].set_ylabel("幅值")
    axes[0].legend(fontsize=9)
    axes[1].plot(tt, avg, color="#c0392b", lw=1.6, label="同步平均后（缺陷冲击被放大）")
    axes[1].set_title("TSA 同步平均：与转频不同步的噪声被平均掉，缺陷清晰")
    axes[1].set_xlabel("时间 (ms)")
    axes[1].set_ylabel("幅值")
    axes[1].legend(fontsize=9)
    plt.tight_layout()
    savefig("advanced2_tsa", subdir="signal")
    plt.close()


def demo_cepstrum():
    """倒频谱：很多等间距边带在频谱里很乱，在倒频域塌成一个峰（间距的倒数）。"""
    set_style()
    fs = 12000
    n = fs
    t = np.arange(n) / fs
    f0 = 1000.0                  # 载波（如齿轮啮合频率）
    fm = 40.0                    # 调制频率（如转频）→ 边带间距 40Hz
    sig = np.zeros(n)
    # 载波 + 多阶边带
    sig += np.sin(2 * np.pi * f0 * t)
    for k in range(1, 6):
        sig += 0.25 / k * np.sin(2 * np.pi * (f0 + k * fm) * t)
        sig += 0.25 / k * np.sin(2 * np.pi * (f0 - k * fm) * t)
    sig += 0.15 * np.random.default_rng(1).normal(0, 1, n)

    sp = np.abs(np.fft.rfft(sig)) / n
    freq = np.fft.rfftfreq(n, d=1 / fs)
    # 真实倒频谱：实信号取 |FFT| 取 log 再 IFFT 取实部；用 quefrency（秒）
    cep = np.fft.irfft(np.log(np.abs(np.fft.rfft(sig)) + 1e-6), n=n)
    quef = np.arange(n) / fs

    # 只画正 quefrency 前段
    qmask = (quef > 0) & (quef < 0.05)
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=False)
    axes[0].plot(freq, sp, color="#2c3e50", lw=0.8)
    axes[0].set_title("频谱：载波 1000Hz 周围一堆 ±40Hz 边带，肉眼难直接读出'间距'")
    axes[0].set_xlabel("频率 (Hz)")
    axes[0].set_ylabel("幅值")
    axes[0].set_xlim(800, 1200)
    axes[1].plot(quef[qmask] * 1e3, np.abs(cep[qmask]), color="#c0392b", lw=1.2)
    # 边带间距 40Hz → 倒频峰在 1/40 = 25ms
    axes[1].axvline(1000 / fm, color="#27ae60", ls="--", lw=1.2,
                    label="1/40Hz = 25 ms（边带间距的确切值）")
    axes[1].set_title("倒频谱：一堆边带塌成一个峰，峰位置=边带间距的倒数")
    axes[1].set_xlabel("倒频率 quefrency (ms)")
    axes[1].set_ylabel("倒谱幅值")
    axes[1].legend(fontsize=9)
    plt.tight_layout()
    savefig("advanced2_cepstrum", subdir="signal")
    plt.close()


def demo_rotor():
    """转子动力学频谱指纹：不平衡 1x / 不对中 2x / 油膜涡动 亚同步。"""
    set_style()
    fs = 2000
    n = 4 * fs
    t = np.arange(n) / fs
    fr = 30.0                    # 转频 30Hz
    rng = np.random.default_rng(2)
    sig = 0.5 * np.sin(2 * np.pi * fr * t)              # 1x 不平衡
    sig += 0.3 * np.sin(2 * np.pi * 2 * fr * t)         # 2x 不对中
    sig += 0.25 * np.sin(2 * np.pi * 0.42 * fr * t)    # 亚同步：油膜涡动 ~0.42x
    sig += 0.2 * np.sin(2 * np.pi * 0.5 * fr * t)      # 半频（松动/碰磨）
    sig += 0.1 * rng.normal(0, 1, n)

    sp = np.abs(np.fft.rfft(sig)) / n
    freq = np.fft.rfftfreq(n, d=1 / fs)
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(freq, sp, color="#2c3e50", lw=0.8)
    # 标注
    ax.axvline(fr, color="#e67e22", ls="--", lw=1.2)
    ax.axvline(2 * fr, color="#2980b9", ls="--", lw=1.2)
    ax.axvline(0.42 * fr, color="#c0392b", ls="--", lw=1.2)
    ax.axvline(0.5 * fr, color="#8e44ad", ls="--", lw=1.2)
    ax.annotate("1× 不平衡", (fr, sp[int(fr)] + 0.02), color="#e67e22", fontsize=10)
    ax.annotate("2× 不对中", (2 * fr, sp[int(2 * fr)] + 0.02), color="#2980b9", fontsize=10)
    ax.annotate("0.42× 油膜涡动(亚同步)", (0.42 * fr, sp[int(0.42 * fr)] + 0.02),
                color="#c0392b", fontsize=10)
    ax.annotate("0.5× 松动/碰磨", (0.5 * fr, sp[int(0.5 * fr)] + 0.02),
                color="#8e44ad", fontsize=10)
    ax.set_title("转子动力学：不同故障在谱上留下不同'指纹'（看 1x / 2x / 亚同步）")
    ax.set_xlabel("频率 (Hz)")
    ax.set_ylabel("幅值")
    ax.set_xlim(0, 120)
    plt.tight_layout()
    savefig("advanced2_rotor", subdir="signal")
    plt.close()


if __name__ == "__main__":
    demo_tsa()
    demo_cepstrum()
    demo_rotor()
    print("done")
