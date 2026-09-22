"""
数据层演示：CWRU 信号的时域、FFT 频谱与包络解调。

对比 Normal 与 InnerRace 故障信号，展示：
- 时域看起来都像噪声（肉眼难分）；
- FFT 频谱里故障能量被宽带振动淹没；
- 包络解调后，内圈故障特征频率 BPFI 处出现明显尖峰。

运行：python src/module_02_signal/demo_fft_envelope.py
产出：figures/signal/cwru_spectrum_envelope.png
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.common.plot_style import set_style, savefig
from src.common.cwru_loader import load_mat
from src.common.signal_utils import (
    fft_spectrum, envelope_spectrum, fault_frequencies_6205, time_features,
)

import matplotlib.pyplot as plt


CWRU_DIR = ROOT / "data" / "CWRU" / "CWRU_data"


def pick_signal(file_name, target_label_hint=None):
    recs = list(load_mat(CWRU_DIR / file_name).values())
    rec = recs[0]
    return rec["signal"], rec["rpm"], rec["key"]


def main():
    set_style()
    # Normal: 97.mat -> X097_DE_time ; InnerRace: 118.mat -> X118_DE_time
    x_norm, rpm_n, k_n = pick_signal("97.mat")
    x_ir, rpm_i, k_i = pick_signal("118.mat")
    rpm = rpm_i or rpm_n or 1797.0
    fs = 12000.0  # CWRU 12k 驱动端采样率

    ff = fault_frequencies_6205(rpm)
    print(f"rpm={rpm}, 故障特征频率: { {k: round(v, 1) for k, v in ff.items()} }")

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))

    # 时域（前 0.03 s）
    n_show = int(0.03 * fs)
    for ax, (x, name) in zip([axes[0, 0], axes[0, 1]], [(x_norm, "Normal"), (x_ir, "InnerRace")]):
        ax.plot(np.arange(n_show) / fs * 1000, x[:n_show], lw=0.6)
        ax.set_title(f"时域（{name}）前 30 ms")
        ax.set_xlabel("时间 (ms)")
        ax.set_ylabel("加速度 (g)")

    # FFT 频谱
    freq_n, mag_n = fft_spectrum(x_norm, fs)
    freq_i, mag_i = fft_spectrum(x_ir, fs)
    axes[1, 0].plot(freq_n, mag_n, color="#4C72B0", lw=0.8, label="Normal")
    axes[1, 0].plot(freq_i, mag_i, color="#C44E52", lw=0.8, label="InnerRace")
    axes[1, 0].set_xlim(0, 5000)
    axes[1, 0].set_title("FFT 频谱（0-5 kHz）")
    axes[1, 0].set_xlabel("频率 (Hz)")
    axes[1, 0].set_ylabel("幅值")
    axes[1, 0].legend()

    # 包络谱
    ef_n, em_n = envelope_spectrum(x_norm, fs)
    ef_i, em_i = envelope_spectrum(x_ir, fs)
    axes[1, 1].plot(ef_n, em_n, color="#4C72B0", lw=0.8, label="Normal")
    axes[1, 1].plot(ef_i, em_i, color="#C44E52", lw=0.8, label="InnerRace")
    for fname, fv in ff.items():
        axes[1, 1].axvline(fv, color="green", ls="--", lw=1)
        axes[1, 1].text(fv, axes[1, 1].get_ylim()[1] * 0.9, f" {fname}={fv:.0f}Hz",
                        color="green", fontsize=8, rotation=90, va="top")
    axes[1, 1].set_xlim(0, 400)
    axes[1, 1].set_title("包络谱（0-400 Hz）：内圈故障 BPFI 处出现尖峰")
    axes[1, 1].set_xlabel("频率 (Hz)")
    axes[1, 1].set_ylabel("包络幅值")
    axes[1, 1].legend()

    fig.suptitle("CWRU 轴承信号：时域几乎一样，包络解调让故障现形", fontsize=14)
    savefig("cwru_spectrum_envelope", subdir="signal")
    plt.close()

    # 顺手打印时域特征，给诊断章做铺垫
    print("时域特征对比：")
    for name, x in [("Normal", x_norm), ("InnerRace", x_ir)]:
        tf = time_features(x)
        print(f"  {name:10s}: RMS={tf['rms']:.4f}, Kurtosis={tf['kurtosis']:.3f}, "
              f"Crest={tf['crest_factor']:.2f}")


if __name__ == "__main__":
    main()
