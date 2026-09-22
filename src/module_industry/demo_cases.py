"""工业案例：风电齿轮箱轴承"从健康到失效"的振动退化趋势 + 早期预警救回抢修。

用一个合成但符合物理直觉的退化曲线，演示 PHM 价值的核心——
在"还能跑、但已开始剥落"的窗口就抓住，把非计划抢修变成计划内更换。

生成：
    figures/industry/case_wind_gearbox_trend.png
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.common.plot_style import set_style, savefig

rng = np.random.default_rng(20260922)
set_style()


def simulate_bearing_degradation(n_weeks=52, seed_onset=30, noise=0.06):
    """合成轴承振动 RMS 退化（相对健康基线 1.0）。

    - 健康期：噪声围绕基线；
    - 起始剥落(seed_onset 周后)：按指数增长（呼应 Paris 定律的"裂纹越长越快"，见 13）；
    - 失效阈值：约定 RMS 超过 2.5 即视为功能失效（配合 ISO 10816 区，见 20）。
    """
    t = np.arange(n_weeks)
    base = 1.0
    # 指数增长分量：从 onset 周开始
    growth = np.where(t >= seed_onset,
                      np.exp(0.11 * (t - seed_onset)), 1.0)
    sig = base * growth * (1.0 + rng.normal(0, noise, n_weeks))
    sig = np.clip(sig, 0.5, None)
    return t, sig


def main():
    t, rms = simulate_bearing_degradation()

    # 关键事件点
    onset_week = 30                      # 物理起始剥落
    fail_week = int(np.argmax(rms >= 2.5))  # 失效点
    alarm_week = onset_week + 6          # 假设 PHM 在 onset 后 6 周发出可行动预警
    # 预警阈值设在失效阈之前，给计划窗口
    warn_thr = 1.7
    warn_week = int(np.argmax(rms >= warn_thr)) if np.any(rms >= warn_thr) else None

    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 左：退化趋势
    ax = axes[0]
    ax.plot(t, rms, color="#1f77b4", lw=1.8, label="振动 RMS(相对健康=1)")
    ax.axhline(2.5, color="#d62728", ls="--", lw=1.2, label="功能失效阈值(2.5)")
    ax.axhline(warn_thr, color="#ff7f0e", ls="--", lw=1.2,
               label=f"预警阈值({warn_thr})")
    if warn_week is not None:
        ax.axvline(warn_week, color="#ff7f0e", alpha=0.5)
        ax.scatter([warn_week], [rms[warn_week]], color="#ff7f0e", zorder=5)
        ax.annotate(f"PHM 预警\n第{warn_week}周", (warn_week, rms[warn_week]),
                    textcoords="offset points", xytext=(8, -28),
                    fontsize=9, color="#ff7f0e")
    ax.axvline(fail_week, color="#d62728", alpha=0.5)
    ax.scatter([fail_week], [rms[fail_week]], color="#d62728", zorder=5)
    ax.annotate(f"功能失效\n第{fail_week}周", (fail_week, rms[fail_week]),
                textcoords="offset points", xytext=(6, 10),
                fontsize=9, color="#d62728")
    ax.set_xlabel("运行周数")
    ax.set_ylabel("振动 RMS(相对值)")
    ax.set_title("风电齿轮箱轴承：PHM 早期预警 vs 突发失效")
    ax.legend(fontsize=8)

    # 右：成本对比（量级示意，单位：万元；海上风电吊船抢修极贵）
    ax = axes[1]
    planned = 18      # 计划内更换（利用吊车窗口）
    unplanned = 120   # 非计划海上抢修（吊船+连锁停机+交期）
    bars = [planned, unplanned]
    labels = ["计划内更换\n(PHM 预警后)", "非计划抢修\n(未预警)"]
    colors = ["#2ca02c", "#d62728"]
    b = ax.bar(labels, bars, color=colors, width=0.55)
    ax.bar_label(b, fmt="%d 万", padding=3, fontsize=11)
    ax.set_ylabel("单次事件成本(万元，量级示意)")
    ax.set_title(f"一次提前 {fail_week - warn_week} 周的预警\n≈ 省 {(unplanned - planned)} 万")
    ax.set_ylim(0, unplanned * 1.15)

    savefig("case_wind_gearbox_trend", subdir="industry")
    print(f"onset={onset_week}  warn={warn_week}  fail={fail_week}  "
          f"lead={fail_week - warn_week if warn_week else 'NA'} weeks")


if __name__ == "__main__":
    main()
