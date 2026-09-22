"""
场景 C：锂电池 SOH / RUL 健康预测（等效电路模型物理仿真）
=========================================================================

真实电池数据（NASA、牛津）体积大、需下载；为了让教程「开箱即跑、可复现」，
这里用一个**物理可解释的等效电路模型(ECM)+ 老化模型**自行生成数据：

  1) Thevenin 模型：V = OCV(SOC) - I·R0 - V1，其中 RC 支路 V1 描述极化。
  2) 老化模型：容量 Q 随累计吞吐(Throughput)线性衰减；内阻 R0 随老化上升。
     SOH = Q / Q_nominal；EOL(寿命终点) 通常定义为 SOH 跌到 80%。

我们要解决的工程问题：
  - 在真实 BMS 里，你只能「量」到电流、端电压、温度，算不出 SOC/容量真值。
  - 于是用「脉冲内阻测试 + 库仑计数」得到带噪声的内阻与容量观测值，
    再训练模型估计 SOH，并外推 RUL（还能撑多少循环到 80%）。

本脚本输出：SOH 衰减曲线、内阻随老化上升、数据驱动的 SOH 估计 + RUL 预测带。

运行：从 PHM 根目录执行
  python src/scenarios/scenario_battery/sim.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib.pyplot as plt

from src.common.plot_style import set_style, savefig
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score


# ----------------------------- 物理模型 -----------------------------
def ocv(soc: np.ndarray) -> np.ndarray:
    """开路电压曲线（典型磷酸铁锂/三元混合的平滑近似，3.0~4.2V）。"""
    soc = np.clip(soc, 0, 1)
    return 3.0 + 1.2 * soc - 0.5 * soc ** 2 + 0.25 * soc ** 3


def simulate_cell(n_cycles=400, q_nominal=2.0, seed=0):
    """
    返回每个循环的「真值」与「观测值」：
      cycle      : 循环序号
      Q_true     : 真实容量(Ah)
      R0_true    : 真实内阻(欧姆)
      SOH_true   : 真实健康度
      R0_meas    : 脉冲测试测得的内阻（含噪声）
      Q_meas     : 库仑计数得到的容量（含噪声）
    老化：每 Ah 吞吐让容量降 0.0008，内阻随 SOH 下降而上升。
    """
    rng = np.random.default_rng(seed)
    dt = 1.0  # s
    Q = q_nominal
    R0 = 0.05
    throughput = 0.0  # 累计吞吐 Ah
    V1 = 0.0
    R1, C1 = 0.02, 30.0  # 极化支路
    tau = R1 * C1

    cycles, Q_true, R0_true = [], [], []
    R0_meas, Q_meas = [], []
    for k in range(n_cycles):
        # 随机充放电剖面（电流有正有负，幅值 0.5~2C）
        soc = 0.5
        I = rng.choice([-1, 1]) * rng.uniform(0.5, 2.0) * q_nominal
        steps = 3600  # 模拟 1 小时片段
        for _ in range(steps):
            dq = I * dt / 3600.0
            soc -= dq / Q
            soc = np.clip(soc, 0.02, 0.98)
            V1 += dt / tau * (I * R1 - V1)
            throughput += abs(dq)
        # 老化更新（基于本循环吞吐）
        Q = max(q_nominal - 0.0008 * throughput, 0.6 * q_nominal)
        R0 = 0.05 + 0.10 * (1 - Q / q_nominal)
        cycles.append(k)
        Q_true.append(Q)
        R0_true.append(R0)
        SOH_true = Q / q_nominal
        # 观测（BMS 能测到的、带噪声）
        R0_meas.append(R0 * rng.normal(1.0, 0.04))
        Q_meas.append(Q * rng.normal(1.0, 0.02))
    out = dict(
        cycle=np.array(cycles),
        Q_true=np.array(Q_true),
        R0_true=np.array(R0_true),
        SOH_true=np.array([q / q_nominal for q in Q_true]),
        R0_meas=np.array(R0_meas),
        Q_meas=np.array(Q_meas),
    )
    return out


def main():
    set_style()
    d = simulate_cell(n_cycles=400)

    # ---- 图1：SOH 衰减 + 内阻上升 ----
    fig, ax1 = plt.subplots()
    ax1.plot(d["cycle"], d["SOH_true"] * 100, color="#1f77b4", label="SOH(真实)")
    ax1.axhline(80, color="r", ls="--", lw=1, label="EOL 阈值 80%")
    ax1.set_xlabel("循环次数")
    ax1.set_ylabel("SOH (%)", color="#1f77b4")
    ax2 = ax1.twinx()
    ax2.plot(d["cycle"], d["R0_true"], color="#ff7f0e", alpha=0.7, label="内阻(真实)")
    ax2.set_ylabel("内阻 (Ω)", color="#ff7f0e")
    ax1.set_title("锂电池老化：SOH 衰减与内阻上升")
    ax1.legend(loc="center right")
    savefig("soh_fade", "scenario_battery")

    # ---- 数据驱动 SOH 估计（用带噪声的观测值）----
    # 真实场景：BMS 只知道 R0_meas / Q_meas，不知道 SOH_true。
    # 关键教训：SOH 与「容量观测值」近似线性（SOH≈Q_meas/Q_nominal），
    # 这种**物理可外推**的关系要用线性模型；随机森林等树模型不会外推，
    # 一旦测试段 cycle/容量超出训练区间就会给出恒定错误预测（R² 变负）。
    from sklearn.linear_model import Ridge
    X = np.column_stack([d["R0_meas"], d["Q_meas"]])
    y = d["SOH_true"]
    # 用前 60% 循环训练，预测后 40%（模拟「早期数据估全寿命」）
    n_tr = int(0.6 * len(y))
    model = Ridge(alpha=0.1)
    model.fit(X[:n_tr], y[:n_tr])
    soh_pred = model.predict(X)

    rmse = float(np.sqrt(mean_squared_error(y[n_tr:], soh_pred[n_tr:])))
    r2 = float(r2_score(y[n_tr:], soh_pred[n_tr:]))
    print(f"[SOH 估计] 验证 RMSE={rmse:.4f}  R²={r2:.4f}（用前{n_tr}循环训练，预测后段）")

    fig, ax = plt.subplots()
    ax.plot(d["cycle"], y * 100, "k-", lw=1.5, label="SOH 真实")
    ax.plot(d["cycle"][:n_tr], soh_pred[:n_tr], "g.", ms=3, label="训练区间估计")
    ax.plot(d["cycle"][n_tr:], soh_pred[n_tr:], "r.", ms=3, label="预测区间估计")
    ax.axvline(d["cycle"][n_tr], color="gray", ls=":", lw=1)
    ax.axhline(80, color="r", ls="--", lw=1)
    ax.set_xlabel("循环次数")
    ax.set_ylabel("SOH (%)")
    ax.set_title("数据驱动 SOH 估计（黑=真值，红=早期模型外推）")
    ax.legend()
    savefig("soh_estimate", "scenario_battery")

    # ---- RUL：预测 SOH 触达 80% 的循环 ----
    def rul_estimate(pred_soh, cycle):
        below = np.where(pred_soh <= 0.80)[0]
        if len(below) == 0:
            return len(cycle)  # 未到 EOL
        return int(below[0] - np.argmax(pred_soh < pred_soh[0]))  # 简化：首个跌破点

    eol_cycle_true = int(np.where(d["SOH_true"] <= 0.80)[0][0])
    # 用训练区间模型对未来 SOH 做线性外推，找 80% 交点
    cyc_future = np.arange(d["cycle"][n_tr], len(d["SOH_true"]) + 100)
    from numpy.polynomial import polynomial as P
    coef = P.polyfit(d["cycle"][:n_tr], soh_pred[:n_tr], 1)
    fut_soh = P.polyval(cyc_future, coef)
    cross = np.where(fut_soh <= 0.80)[0]
    eol_pred = int(cyc_future[cross[0]]) if len(cross) else len(d["cycle"])
    print(f"[RUL] 真实 EOL 循环≈{eol_cycle_true}；模型外推 EOL≈{eol_pred}；"
          f"误差 {abs(eol_pred - eol_cycle_true)} 循环")

    fig, ax = plt.subplots()
    ax.plot(d["cycle"], d["SOH_true"] * 100, "k-", label="SOH 真实")
    ax.plot(cyc_future, fut_soh * 100, "r--", label="早期模型外推")
    ax.axhline(80, color="r", ls=":", lw=1)
    ax.axvline(eol_cycle_true, color="k", ls="--", label=f"真实 EOL≈{eol_cycle_true}")
    ax.axvline(eol_pred, color="g", ls="--", label=f"预测 EOL≈{eol_pred}")
    ax.set_xlabel("循环次数")
    ax.set_ylabel("SOH (%)")
    ax.set_title("RUL：从早期数据外推寿命终点")
    ax.legend()
    savefig("rul_battery", "scenario_battery")


if __name__ == "__main__":
    main()
