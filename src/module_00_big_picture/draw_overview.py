"""
生成 PHM 技术全景图：感知 -> 采集 -> 处理 -> 诊断 -> 预测 -> 决策 -> 反馈。

运行：python src/module_00_big_picture/draw_overview.py
产出：figures/big_picture/phm_loop.png
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.common.plot_style import set_style, savefig

import matplotlib.pyplot as plt


def draw():
    set_style()
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.axis("off")

    stages = [
        ("感知层\n传感器/执行器", 0.06, "#4C72B0"),
        ("采集层\nDAQ/ADC/IEPE", 0.22, "#55A868"),
        ("计算/边缘云\n信号处理+特征", 0.38, "#C44E52"),
        ("诊断\n故障分类/定位", 0.54, "#8172B3"),
        ("预测\nRUL/退化建模", 0.70, "#CCB974"),
        ("决策\n维护/告警", 0.86, "#17BECF"),
    ]
    y = 0.55
    for stage in stages:
        x = stage[1]
        color = stage[2]
        ax.add_patch(plt.Rectangle((x, y - 0.12), 0.12, 0.24, color=color, alpha=0.85))
        ax.text(x + 0.06, y, stage[0], ha="center", va="center", color="white",
                fontsize=11, fontweight="bold")

    # 正向箭头
    for i in range(len(stages) - 1):
        x0 = stages[i][1] + 0.12
        x1 = stages[i + 1][1]
        ax.annotate("", xy=(x1, y), xytext=(x0, y),
                    arrowprops=dict(arrowstyle="->", lw=2, color="gray"))

    # 反馈环（决策 -> 感知）
    ax.annotate("", xy=(stages[0][1] + 0.02, y - 0.16),
                xytext=(stages[-1][1] + 0.10, y - 0.16),
                arrowprops=dict(arrowstyle="->", lw=2, color="orange",
                                connectionstyle="arc3,rad=-0.3"))
    ax.text(0.46, y - 0.22, "维护执行 / 数据回流（闭环优化）", ha="center",
            color="orange", fontsize=10)

    ax.text(0.5, 0.92, "PHM（故障预测与健康管理）技术全景", ha="center",
            fontsize=16, fontweight="bold")
    ax.text(0.5, 0.84, "从一根振动信号到一次精准的维护决策", ha="center",
            fontsize=11, color="gray")

    # 底栏关键词
    keywords = [
        "硬件：加速度/温度/电流/麦克风",
        "软件：FFT/包络/小波/特征",
        "算法：SVM·RF·1D-CNN·LSTM·Transformer",
        "落地：边缘推理·云端训练·数字孪生",
    ]
    for i, kw in enumerate(keywords):
        ax.text(0.02 + i * 0.25, 0.18, kw, ha="left", va="center",
                fontsize=9, color="#333333",
                bbox=dict(boxstyle="round,pad=0.3", fc="#F2F2F2", ec="#CCCCCC"))

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    savefig("phm_loop", subdir="big_picture")
    plt.close()


if __name__ == "__main__":
    draw()
