"""
概念示意图生成器（给初学者建立直觉）
=========================================================================
本脚本不依赖真实数据，用 matplotlib 画出 PHM 里最该「先有个画面」的概念图，
配合各模块文档里的真实数据图一起看，初学者更容易建立整体认知：

  1) bearing_geometry   : 滚动轴承剖面 + 内/外圈/滚动体缺陷位置 + 故障频率
  2) signal_chain       : 传感器→IEPE调理→ADC→处理的信号链
  3) envelope_concept   : 包络解调原理（冲击→调幅→边带→包络谱露出故障频率）
  4) degradation_rul    : 退化曲线 + 健康阈值 + RUL 箭头
  5) cnn_arch           : 1D-CNN 结构示意
  6) gear_schematic     : 齿轮啮合 + 故障齿 + GMF/边带
  7) deploy_arch        : 边缘/云 部署架构

全部落盘到 figures/illustrations/。

运行：从 PHM 根目录执行
  python src/module_illustrations/draw_concepts.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, FancyArrowPatch, Arc
from scipy.signal import hilbert

from src.common.plot_style import set_style, savefig


# ----------------------------- 1. 轴承几何 -----------------------------
def bearing_geometry():
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect("equal")
    # 外圈、内圈、节圆
    ax.add_patch(Circle((0, 0), 1.0, fill=False, color="#444", lw=3))   # 外圈
    ax.add_patch(Circle((0, 0), 0.45, fill=False, color="#444", lw=3))  # 内圈
    ax.add_patch(Circle((0, 0), 0.72, ls="--", fill=False, color="#888"))  # 节圆
    # 滚动体
    n_balls = 9
    for i in range(n_balls):
        th = 2 * np.pi * i / n_balls
        bx, by = 0.72 * np.cos(th), 0.72 * np.sin(th)
        ax.add_patch(Circle((bx, by), 0.16, color="#1f77b4"))
    # 缺陷标记
    ax.plot([0.72, 1.0], [0, 0], "r*", ms=18)          # 外圈缺陷
    ax.plot([0, 0.45], [0, 0], "r*", ms=18)            # 内圈缺陷
    ax.plot([0.72 * np.cos(0.3), 0.72 * np.cos(0.3) + 0.16 * np.cos(0.3)],
            [0.72 * np.sin(0.3), 0.72 * np.sin(0.3) + 0.16 * np.sin(0.3)],
            "r*", ms=16)                                # 滚动体缺陷
    ax.text(0, -1.25, "外圈 Outer Race", ha="center", color="r")
    ax.text(0, 0.62, "内圈 Inner Race", ha="center", color="r")
    ax.text(0.9, 0.9, "滚动体 Ball", color="#1f77b4")
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.35, 1.3)
    ax.axis("off")
    ax.set_title("滚动轴承剖面与缺陷位置（红★=缺陷）")
    # 故障频率注释框
    ax.text(0.0, -1.05, "", fontsize=9)
    fig.text(0.5, 0.02,
             "故障特征频率：BPFI(内圈)≈162Hz  BPFO(外圈)≈107Hz  BSF(滚动体)≈71Hz  FTF(保持架)≈12Hz  (rpm≈1796)",
             ha="center", fontsize=9, color="#333")
    savefig("bearing_geometry", "illustrations")


# ----------------------------- 2. 信号链 -----------------------------
def signal_chain():
    fig, ax = plt.subplots(figsize=(11, 3.5))
    ax.axis("off")
    nodes = [
        ("振动加速度\n传感器(IEPE)", 0.06),
        ("信号调理\n(恒流源/放大)", 0.27),
        ("抗混叠滤波\n+ ADC采样", 0.5),
        ("数字信号\n处理", 0.72),
        ("特征/模型\n+ 诊断决策", 0.92),
    ]
    for i, (txt, x) in enumerate(nodes):
        ax.add_patch(Rectangle((x - 0.07, 0.35), 0.14, 0.3, color="#1f77b4", alpha=0.85))
        ax.text(x, 0.5, txt, ha="center", va="center", color="white", fontsize=9)
        if i < len(nodes) - 1:
            ax.annotate("", xy=(nodes[i + 1][1] - 0.07, 0.5), xytext=(x + 0.07, 0.5),
                        arrowprops=dict(arrowstyle="->", color="#333", lw=1.5))
    # 顶部画信号形态演变
    t = np.linspace(0, 1, 200)
    ax.plot(t[:60], 0.12 + 0.08 * np.sin(2 * np.pi * 8 * t[:60]), color="#d62728")  # 模拟波形
    ax.text(0.13, 0.08, "模拟振动", ha="center", fontsize=8, color="#d62728")
    ax.text(0.5, 0.08, "数字序列(采样)", ha="center", fontsize=8)
    ax.text(0.92, 0.08, "故障类别/健康度", ha="center", fontsize=8, color="#2ca02c")
    ax.set_title("PHM 信号链：从物理振动到诊断决策")
    savefig("signal_chain", "illustrations")


# ----------------------------- 3. 包络解调原理 -----------------------------
def envelope_concept():
    fs = 12000.0
    dur = 0.05
    t = np.arange(int(fs * dur)) / fs
    fc = 2000.0          # 载波（共振频率附近）
    f_bp = 162.0         # 故障特征频率（BPFI 量级）
    env = 1 + 0.9 * np.cos(2 * np.pi * f_bp * t)   # 冲击造成的幅值调制
    x = env * np.sin(2 * np.pi * fc * t)
    x += 0.05 * np.random.randn(len(t))

    fig, axs = plt.subplots(2, 2, figsize=(11, 6))
    axs[0, 0].plot(t[:600], x[:600], color="#1f77b4")
    axs[0, 0].set_title("① 原始振动（被故障周期性调幅）")
    axs[0, 0].set_xlabel("时间")
    # 频谱（看载波与边带）
    freq, mag = _fft(x, fs)
    band = (freq >= fc - 400) & (freq <= fc + 400)
    axs[0, 1].plot(freq[band], mag[band], color="#ff7f0e")
    axs[0, 1].axvline(fc, color="k", ls="--", lw=1)
    axs[0, 1].axvline(fc - f_bp, color="r", ls=":", lw=1)
    axs[0, 1].axvline(fc + f_bp, color="r", ls=":", lw=1)
    axs[0, 1].set_title("② 频谱：载波fc两侧出现边带fc±f_bp")
    axs[0, 1].set_xlabel("频率 (Hz)")
    # 包络
    env_sig = np.abs(hilbert(x))
    axs[1, 0].plot(t[:600], env_sig[:600], color="#2ca02c")
    axs[1, 0].set_title("③ 包络信号（解调出低频调制）")
    axs[1, 0].set_xlabel("时间")
    # 包络谱（故障频率露出来）
    freq_e, mag_e = _fft(env_sig, fs)
    b = (freq_e >= 0) & (freq_e <= 400)
    axs[1, 1].plot(freq_e[b], mag_e[b], color="#d62728")
    axs[1, 1].axvline(f_bp, color="k", ls="--", lw=1)
    axs[1, 1].set_title("④ 包络谱：在故障频率 f_bp 出现尖峰")
    axs[1, 1].set_xlabel("频率 (Hz)")
    fig.suptitle("包络解调原理：故障冲击→幅值调制→边带→包络谱露出故障频率", fontsize=13)
    savefig("envelope_concept", "illustrations")


def _fft(x, fs):
    X = np.fft.rfft(x - x.mean())
    freq = np.fft.rfftfreq(len(x), 1 / fs)
    return freq, np.abs(X)


# ----------------------------- 4. 退化与 RUL -----------------------------
def degradation_rul():
    fig, ax = plt.subplots(figsize=(10, 4.5))
    t = np.arange(0, 101)
    # 退化：初期平缓、后期加速（典型磨损）
    health = 100 - 100 * (1 - np.exp(-t / 45)) * 0.85 - t * 0.12
    health = np.clip(health, 0, 100)
    ax.plot(t, health, color="#1f77b4", lw=2, label="设备健康指标")
    th = 80
    ax.axhline(th, color="r", ls="--", lw=1.5, label="失效阈值 (80%)")
    # 当前时刻
    now = 55
    ax.scatter([now], [health[now]], color="k", zorder=5)
    ax.annotate("当前时刻", (now, health[now]), xytext=(now - 18, health[now] + 8),
                arrowprops=dict(arrowstyle="->"))
    # RUL 箭头
    eol = int(np.where(health <= th)[0][0])
    ax.annotate("", xy=(eol, th), xytext=(now, health[now]),
                arrowprops=dict(arrowstyle="->", color="green", lw=2))
    ax.text((now + eol) / 2, (health[now] + th) / 2 + 4,
            f"RUL ≈ {eol - now} 单位时间", color="green", fontsize=11, ha="center")
    ax.set_xlabel("运行时间 / 循环")
    ax.set_ylabel("健康指标 (%)")
    ax.set_title("剩余使用寿命 RUL：从当前到越过失效阈值的距离")
    ax.legend()
    savefig("degradation_rul", "illustrations")


# ----------------------------- 5. 1D-CNN 结构 -----------------------------
def cnn_arch():
    fig, ax = plt.subplots(figsize=(11, 3))
    ax.axis("off")
    blocks = [
        ("输入\n1×2048\n振动窗口", "#1f77b4"),
        ("Conv1D\n16通道, k=7\n+ReLU", "#ff7f0e"),
        ("MaxPool\n×2", "#ff7f0e"),
        ("Conv1D\n32通道, k=5\n+ReLU", "#ff7f0e"),
        ("Conv1D\n64通道, k=3\n+ReLU", "#ff7f0e"),
        ("全局池化\n+ Flatten", "#2ca02c"),
        ("全连接\n64 → 4", "#d62728"),
        ("输出\n4类概率", "#d62728"),
    ]
    n = len(blocks)
    for i, (txt, c) in enumerate(blocks):
        x = 0.02 + i * (0.96 / n)
        ax.add_patch(Rectangle((x, 0.3), 0.96 / n - 0.01, 0.4, color=c, alpha=0.85))
        ax.text(x + (0.96 / n - 0.01) / 2, 0.5, txt, ha="center", va="center",
                color="white", fontsize=8)
        if i < n - 1:
            ax.annotate("", xy=(x + 0.96 / n, 0.5), xytext=(x + 0.96 / n - 0.01, 0.5),
                        arrowprops=dict(arrowstyle="->", color="#333"))
    ax.set_title("1D-CNN 故障诊断结构：直接在原始波形上学特征")
    savefig("cnn_arch", "illustrations")


# ----------------------------- 6. 齿轮啮合 -----------------------------
def gear_schematic():
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.axis("off")
    # 画两个齿轮（用多边形齿近似）
    def gear(cx, cy, R, teeth, fault_idx=None, color="#1f77b4"):
        pts = []
        for i in range(teeth * 2):
            ang = np.pi * i / teeth
            r = R if i % 2 == 0 else R * 0.82
            pts.append((cx + r * np.cos(ang), cy + r * np.sin(ang)))
        poly = plt.Polygon(pts, closed=True, color=color, alpha=0.85)
        ax.add_patch(poly)
        ax.add_patch(Circle((cx, cy), R * 0.25, color="white"))
        if fault_idx is not None:
            ang = np.pi * (2 * fault_idx) / teeth
            fx, fy = cx + R * np.cos(ang), cy + R * np.sin(ang)
            ax.plot(fx, fy, "r*", ms=22)
    gear(0.35, 0.5, 0.28, 14, fault_idx=3)
    gear(0.75, 0.5, 0.18, 9)
    ax.text(0.35, 0.16, "齿轮A (Z=14)", ha="center")
    ax.text(0.75, 0.24, "齿轮B (Z=9)", ha="center")
    ax.text(0.35, 0.86, "红★=断齿/剥落故障", ha="center", color="r")
    ax.text(0.5, 0.95, "啮合频率 GMF = Z × 轴转频；故障→GMF 两侧出现轴频间隔边带",
            ha="center", fontsize=10, color="#333")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    savefig("gear_schematic", "illustrations")


# ----------------------------- 7. 部署架构 -----------------------------
def deploy_arch():
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.axis("off")
    # 设备层
    ax.add_patch(Rectangle((0.02, 0.55), 0.22, 0.3, color="#1f77b4", alpha=0.85))
    ax.text(0.13, 0.7, "传感器+MCU\n(边缘设备)", ha="center", va="center", color="white", fontsize=9)
    # 网关/推理
    ax.add_patch(Rectangle((0.4, 0.55), 0.22, 0.3, color="#ff7f0e", alpha=0.85))
    ax.text(0.51, 0.7, "边缘网关\nONNX-Runtime", ha="center", va="center", color="white", fontsize=9)
    # 云
    ax.add_patch(Rectangle((0.78, 0.55), 0.2, 0.3, color="#2ca02c", alpha=0.85))
    ax.text(0.88, 0.7, "云平台\n训练/重训", ha="center", va="center", color="white", fontsize=9)
    for a, b in [(0.24, 0.4), (0.62, 0.78)]:
        ax.annotate("", xy=(b, 0.7), xytext=(a, 0.7),
                    arrowprops=dict(arrowstyle="->", color="#333", lw=1.5))
    ax.text(0.31, 0.78, "特征/原始波形", fontsize=8)
    ax.text(0.69, 0.78, "诊断结果", fontsize=8)
    # 回环：云→设备
    ax.annotate("", xy=(0.13, 0.55), xytext=(0.88, 0.3),
                arrowprops=dict(arrowstyle="->", color="#888", lw=1.2, ls="--"))
    ax.text(0.5, 0.32, "模型下发 / 数据回传（持续监控与重训）", ha="center", fontsize=8, color="#555")
    ax.set_title("PHM 部署架构：边缘推理 + 云端持续学习")
    savefig("deploy_arch", "illustrations")


def main():
    set_style()
    bearing_geometry()
    signal_chain()
    envelope_concept()
    degradation_rul()
    cnn_arch()
    gear_schematic()
    deploy_arch()
    phm_lifecycle()
    print("[done] 概念示意图已生成到 figures/illustrations/")


# ------------------------- 8. 工程全生命周期 -------------------------
def phm_lifecycle():
    """把算法之外的工程闭环画出来：算法只是其中一环。"""
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.axis("off")
    nodes = [
        ("① 采集\n传感器+DAQ\n采样率/标定/布置", 0.04, "#1f77b4"),
        ("② 边缘预处理\n滤波/特征/异常初筛\n算力受限", 0.24, "#ff7f0e"),
        ("③ 传输\nMQTT/OPC-UA\n压缩/加密/断网缓存", 0.46, "#9467bd"),
        ("④ 云平台\n数据湖/标注/训练/重训", 0.68, "#2ca02c"),
        ("⑤ 边缘推理\nONNX-RT/INT8\n低延迟", 0.86, "#d62728"),
    ]
    for i, (txt, x, c) in enumerate(nodes):
        ax.add_patch(Rectangle((x, 0.45), 0.15, 0.3, color=c, alpha=0.9))
        ax.text(x + 0.075, 0.6, txt, ha="center", va="center",
                color="white", fontsize=8.5)
        if i < len(nodes) - 1:
            ax.annotate("", xy=(nodes[i + 1][1], 0.6), xytext=(x + 0.15, 0.6),
                        arrowprops=dict(arrowstyle="->", color="#333", lw=1.6))
    # 回环：云→边缘 模型下发；边缘→云 数据回传
    ax.annotate("", xy=(0.86, 0.45), xytext=(0.68, 0.45),
                arrowprops=dict(arrowstyle="->", color="#555", ls="--"))
    ax.text(0.77, 0.41, "模型下发 (OTA)", ha="center", fontsize=7.5, color="#555")
    ax.annotate("", xy=(0.46, 0.45), xytext=(0.24, 0.45),
                arrowprops=dict(arrowstyle="->", color="#555", ls="--"))
    ax.text(0.35, 0.41, "原始/特征回传", ha="center", fontsize=7.5, color="#555")
    # ⑥ 告警 → 工单
    ax.add_patch(Rectangle((0.86, 0.1), 0.15, 0.22, color="#8c564b", alpha=0.9))
    ax.text(0.935, 0.21, "⑥ 告警/工单\nCMMS/SCADA\n误报治理", ha="center", va="center",
            color="white", fontsize=8)
    ax.annotate("", xy=(0.935, 0.1), xytext=(0.935, 0.45),
                arrowprops=dict(arrowstyle="->", color="#8c564b", lw=1.4))
    # 底部红色字：每个环节的真实工程坑
    notes = ["数据质量：丢点/漂移/时钟同步", "算力边界：能跑 CNN 吗",
             "带宽成本 + 安全", "标注稀缺 / 概念漂移", "误报 vs 漏报代价不对称"]
    for txt, x in zip(notes, [0.04, 0.24, 0.46, 0.68, 0.86]):
        ax.text(x + 0.075, 0.36, txt, ha="center", va="top",
                fontsize=7, color="#a00000")
    ax.set_title("PHM 工程全生命周期：算法只是其中一环（红字=每个环节的真实工程坑）",
                 fontsize=12)
    savefig("phm_lifecycle", "illustrations")


if __name__ == "__main__":
    main()
