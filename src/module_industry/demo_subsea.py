"""海缆 / 铺缆船 PHM 直观图：① DAS 沿缆振动入侵定位 ② 海缆故障成因(ICPC 量级)。

演示：
  - 通信海缆的 PHM 重点是"在自有光纤上用 DAS 感知外力 + 定位第三方入侵(锚/渔网)"，
    而不是传统退化 RUL；
  - 故障成因绝大多数是突发外因(锚害+渔业占 65~75%，ICPC 统计)。

生成：figures/industry/subsea_cable_phm.png
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.common.plot_style import set_style, savefig

rng = np.random.default_rng(20260923)
set_style()


def main():
    import matplotlib.pyplot as plt

    # ---- 面板1：DAS 沿缆振动幅值 vs 距离，定位一次第三方入侵 ----
    L = 100.0                      # 海缆长度 km
    x = np.linspace(0, L, 600)
    base = 0.08 + rng.normal(0, 0.02, x.size)   # 环境本底微振动
    amp = base.copy()
    # 在 60~64 km 处模拟一次"锚拖/渔网勾拽"入侵事件
    intr = (x >= 60.0) & (x <= 64.0)
    amp[intr] += 0.9 + rng.normal(0, 0.08, intr.sum())
    loc = x[intr].mean()           # 定位中心

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.plot(x, amp, color="#1f77b4", lw=1.4, label="沿缆振动幅值(DAS)")
    ax.axvspan(60, 64, color="#d62728", alpha=0.15)
    ax.axvline(loc, color="#d62728", ls="--", lw=1.2)
    ax.scatter([loc], [amp[intr].max()], color="#d62728", zorder=5)
    ax.annotate(f"第三方入侵定位\n≈ {loc:.0f} km (锚害/拖网)",
                (loc, amp[intr].max()), textcoords="offset points",
                xytext=(6, 6), fontsize=9, color="#d62728")
    ax.set_xlabel("沿缆距离 (km)")
    ax.set_ylabel("振动幅值 (a.u.)")
    ax.set_title("DAS：海缆自带光纤感知外力并定位入侵")
    ax.legend(fontsize=8)

    # ---- 面板2：海缆故障成因(ICPC 量级示意) ----
    ax = axes[1]
    causes = ["锚泊+渔业\n(第三方)", "自然现象\n(滑坡/海流)", "电缆部件\n失效", "原因不明"]
    vals = [70, 8, 5, 17]          # ICPC 量级(锚+渔合计 65~75%，此处取 70)
    colors = ["#d62728", "#1f77b4", "#2ca02c", "#7f7f7f"]
    b = ax.bar(causes, vals, color=colors, width=0.6)
    ax.bar_label(b, fmt="%d%%", padding=3, fontsize=11)
    ax.set_ylabel("故障占比 (ICPC 量级示意)")
    ax.set_title("海缆故障成因：绝大多数是突发外因")
    ax.set_ylim(0, 85)

    savefig("subsea_cable_phm", subdir="industry")
    print(f"intrusion located at {loc:.1f} km; fault-cause ICPC 量级: 锚+渔=70%")


if __name__ == "__main__":
    main()
