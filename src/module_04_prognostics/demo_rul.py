"""
第 4 层 - 预测（RUL，剩余有用寿命）。

用可复现的退化仿真数据讲清 RUL：
- 生成多条"设备从健康到失效"的健康指标(HI)曲线；
- 基线法：对历史 HI 做指数拟合外推，得到点估计 RUL；
- 深度法：LSTM 序列回归直接预测 RUL；
- 用 MC Dropout 给出预测置信区间（工程里比单点更重要）。

运行：python src/module_04_prognostics/demo_rul.py
产出：figures/prognostics/{rul_curves,rul_scatter,rul_confidence}.png
"""

import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.common.plot_style import set_style, savefig

import matplotlib.pyplot as plt

SEED = 42
THRESH = 0.5          # 失效阈值（HI 低于此值判为失效）
W = 30                # 输入窗口长度（用最近 W 个周期预测）
HIDDEN = 32
EPOCHS = 40
BATCH = 64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def generate_units(n, t_min=80, t_max=200):
    """生成 n 条 run-to-failure 曲线：HI = exp(-k t) + 噪声。"""
    units = []
    for _ in range(n):
        T = int(np.random.randint(t_min, t_max + 1))
        k = np.random.uniform(0.012, 0.05)
        t = np.arange(T)
        h = np.exp(-k * t)
        h_obs = np.clip(h + np.random.normal(0, 0.02, size=T), 0.02, 1.0)
        units.append({"h": h_obs.astype(np.float32), "T": T})
    return units


def make_samples(units):
    X, Y = [], []
    for u in units:
        h = u["h"]
        T = len(h)
        for t in range(W, T):
            X.append(h[t - W:t])
            Y.append(T - t)          # 剩余周期 = 真实 RUL
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)


def baseline_rul(X):
    """模型法：对每条历史窗口做 log 线性拟合，外推到阈值得到 RUL。"""
    preds = []
    for x in X:
        tau = np.arange(len(x))
        slope, _ = np.polyfit(tau, np.log(x + 1e-6), 1)   # log(h) ≈ -k*tau
        k = -slope
        if k <= 0:
            preds.append(250.0)
            continue
        rul = (-np.log(THRESH) / k) - (len(x) - 1)
        preds.append(float(np.clip(rul, 0.0, 250.0)))
    return np.array(preds)


class RULDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return torch.from_numpy(self.X[i]).unsqueeze(-1), torch.tensor(self.Y[i])


class LSTM(nn.Module):
    def __init__(self, hidden=HIDDEN):
        super().__init__()
        self.lstm = nn.LSTM(1, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, 1)
        self.drop = nn.Dropout(0.3)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(self.drop(out[:, -1]))


def train_eval(train_u, test_u):
    Xtr, Ytr = make_samples(train_u)
    Xte, Yte = make_samples(test_u)
    loader = DataLoader(RULDataset(Xtr, Ytr), batch_size=BATCH, shuffle=True, num_workers=2)
    model = LSTM().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.MSELoss()
    for _ in range(EPOCHS):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            crit(model(xb).squeeze(1), yb).backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        yp = model(torch.from_numpy(Xte).unsqueeze(-1).to(DEVICE)).squeeze(1).cpu().numpy()
    rmse = float(np.sqrt(mean_squared_error(Yte, yp)))
    return model, Xte, Yte, yp, rmse


def mc_dropout_rul(model, X, n=30):
    """MC Dropout：开 dropout 多次采样，得到均值与标准差。"""
    model.train()
    Xt = torch.from_numpy(X).to(DEVICE)
    preds = []
    with torch.no_grad():
        for _ in range(n):
            preds.append(model(Xt).squeeze(1).cpu().numpy())
    preds = np.stack(preds, 0)
    return preds.mean(0), preds.std(0)


def main():
    set_style()
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    allu = generate_units(240)
    idx = np.arange(len(allu))
    np.random.shuffle(idx)
    ntr = int(0.7 * len(allu))
    train_u = [allu[i] for i in idx[:ntr]]
    test_u = [allu[i] for i in idx[ntr:]]

    # 图1：若干退化曲线
    fig, ax = plt.subplots(figsize=(8, 4))
    for u in train_u[:6]:
        ax.plot(u["h"], lw=1)
    ax.axhline(THRESH, color="red", ls="--", lw=1, label=f"失效阈值 {THRESH}")
    ax.set_xlabel("运行周期")
    ax.set_ylabel("健康指标 HI")
    ax.set_title("设备退化仿真：HI 从 1 单调下降到阈值即失效")
    ax.legend()
    savefig("rul_curves", subdir="prognostics")
    plt.close()

    # 模型法基线
    Xte = make_samples(test_u)[0]
    yte = make_samples(test_u)[1]
    base = baseline_rul(Xte)

    # LSTM
    model, Xte, yte, yp, rmse = train_eval(train_u, test_u)
    lstm_rmse = rmse

    # 图2：预测 vs 真实 散点
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, (pred, name, c) in zip(axes, [(base, "模型法(指数拟合)", "#55A868"),
                                          (yp, "LSTM", "#4C72B0")]):
        ax.scatter(yte, pred, s=8, alpha=0.4, color=c)
        lim = [0, max(yte.max(), pred.max()) * 1.05]
        ax.plot(lim, lim, "k--", lw=1)
        r = float(np.sqrt(mean_squared_error(yte, pred)))
        ax.set_title(f"{name}  (RMSE={r:.1f})")
        ax.set_xlabel("真实 RUL")
        ax.set_ylabel("预测 RUL")
    fig.suptitle("RUL 预测：真实 vs 预测（测试集）", fontsize=14)
    savefig("rul_scatter", subdir="prognostics")
    plt.close()

    # 图3：单台设备的 RUL 预测 + 置信区间（随时间推进）
    fig, ax = plt.subplots(figsize=(8, 4.5))
    u = test_u[0]
    h = u["h"]
    T = len(h)
    est_mean, est_std = [], []
    t_axis = list(range(W, T))
    for t in t_axis:
        x = h[t - W:t][None, :, None].astype(np.float32)
        m, s = mc_dropout_rul(model, x, n=30)
        est_mean.append(m[0]); est_std.append(s[0])
    est_mean = np.array(est_mean); est_std = np.array(est_std)
    true_rul = np.array([T - t for t in t_axis])
    ax.plot(t_axis, true_rul, "k-", lw=2, label="真实 RUL")
    ax.plot(t_axis, est_mean, color="#4C72B0", label="LSTM 预测")
    ax.fill_between(t_axis, est_mean - 2 * est_std, est_mean + 2 * est_std,
                    color="#4C72B0", alpha=0.2, label="95% 置信区间")
    ax.set_xlabel("当前运行周期")
    ax.set_ylabel("剩余寿命 RUL (周期)")
    ax.set_title("单台设备 RUL 预测随时间的演变（含不确定性）")
    ax.legend()
    savefig("rul_confidence", subdir="prognostics")
    plt.close()

    print(f"[完成] 模型法 RMSE={float(np.sqrt(mean_squared_error(yte, base))):.2f}，"
          f"LSTM RMSE={lstm_rmse:.2f}")
    print("图已保存到 figures/prognostics/")


if __name__ == "__main__":
    main()
