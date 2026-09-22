"""
硬件层演示 1：采样定理与混叠（Aliasing）仿真。

一个真实物理信号（5 Hz 正弦）被不同采样率采样后：
- 采样率 >= 2*信号频率（奈奎斯特）时，能还原；
- 采样率 < 2*信号频率时，高频被“折叠”成低频，产生混叠。

运行：python src/module_01_hardware/demo_sampling_aliasing.py
产出：figures/hardware/sampling_aliasing.png
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.common.plot_style import set_style, savefig

import matplotlib.pyplot as plt


def main():
    set_style()

    f_signal = 5.0          # 真实信号频率 5 Hz
    t = np.linspace(0, 1, 1000, endpoint=False)  # 连续时间轴
    x_true = np.sin(2 * np.pi * f_signal * t)

    sampling_rates = [20.0, 11.0, 8.0]  # 20: 正常; 11: 略低于奈奎斯特(10Hz); 8: 明显混叠
    colors = ["#4C72B0", "#55A868", "#C44E52"]

    fig, axes = plt.subplots(2, len(sampling_rates), figsize=(13, 7))

    for col, (fs, color) in enumerate(zip(sampling_rates, colors)):
        # 采样点
        ts = np.arange(0, 1, 1 / fs)
        xs = np.sin(2 * np.pi * f_signal * ts)

        # 上排：连续信号 + 采样点
        ax = axes[0, col]
        ax.plot(t, x_true, color="gray", lw=1.2, label="真实连续信号 (5 Hz)")
        ax.stem(ts, xs, linefmt=color, markerfmt="o", basefmt=" ", label=f"采样 {fs:.0f} Hz")
        ax.set_title(f"采样率 fs = {fs:.0f} Hz")
        ax.set_xlabel("时间 (s)")
        ax.set_ylabel("幅值")
        ax.legend(fontsize=8)

        # 下排：FFT 频谱（含混叠折叠）
        ax = axes[1, col]
        n = len(xs)
        freq = np.fft.rfftfreq(n, d=1 / fs)
        spec = np.abs(np.fft.rfft(xs)) / n
        ax.stem(freq, spec, linefmt=color, markerfmt="o", basefmt=" ")
        ax.axvline(f_signal, color="gray", ls="--", lw=1, label="真实 5 Hz")
        nyq = fs / 2
        ax.axvline(nyq, color="orange", ls=":", lw=1, label=f"奈奎斯特 {nyq:.0f} Hz")
        ax.set_title(f"频谱 (fs={fs:.0f} Hz)")
        ax.set_xlabel("频率 (Hz)")
        ax.set_ylabel("幅值")
        ax.set_xlim(0, min(fs, 25))
        ax.legend(fontsize=8)

    fig.suptitle("采样定理与混叠仿真：同一 5 Hz 信号，不同采样率下的表现", fontsize=14)
    savefig("sampling_aliasing", subdir="hardware")
    plt.close()


if __name__ == "__main__":
    main()
