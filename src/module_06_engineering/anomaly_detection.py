"""
模块 06：工程实战（一）—— 无监督异常检测：只用「正常」就能抓异常
=========================================================================

真实现场最大的痛点（也是你点出的核心）：**异常样本极少、甚至没有标签**。
你不可能等机器真坏了、再收集几百条故障样本才去训练一个分类器。工业界真正
常用的是「单类 / 无监督」思路：

    只用「正常运行」数据训练一个自编码器（Autoencoder），
    让模型学会「正常长什么样」；
    线上来一段新信号 → 模型重构它 → 重构误差大 = 偏离正常 = 异常。

这天然解决了「异常样本建立难」：故障样本根本不进训练。

本脚本用 CWRU 数据演示端到端：
  1) 只用 Normal 类窗口训练 1D-CNN 自编码器；
  2) 测试时同时给 Normal（应低误差）与 各类故障窗口（应高误差）；
  3) 用 Normal 训练误差的 95 百分位作阈值，报告误报率/检出率与 AUC；
  4) 画图：重构误差分布 / ROC 曲线 / 故障波形重构对比。

运行（从 PHM 根目录）：
  python src/module_06_engineering/anomaly_detection.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve

from src.common.plot_style import set_style, savefig
from src.common.cwru_loader import load_dataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WIN = 2048


class AE(nn.Module):
    """1D-CNN 自编码器：编码下采样到瓶颈向量，解码上采样回原始长度。"""
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv1d(1, 16, 7, padding=3), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(16, 32, 5, padding=2), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool1d(2),
            nn.Flatten(), nn.Linear(64 * (WIN // 8), 64),
        )
        self.dec = nn.Sequential(
            nn.Linear(64, 64 * (WIN // 8)), nn.Unflatten(1, (64, WIN // 8)),
            nn.Conv1d(64, 64, 3, padding=1), nn.ReLU(), nn.Upsample(scale_factor=2),
            nn.Conv1d(64, 32, 3, padding=1), nn.ReLU(), nn.Upsample(scale_factor=2),
            nn.Conv1d(32, 16, 3, padding=1), nn.ReLU(), nn.Upsample(scale_factor=2),
            nn.Conv1d(16, 1, 3, padding=1),
        )

    def forward(self, x):
        return self.dec(self.enc(x))


def make_windows(records, keep_labels, n_per=30, seed=0):
    rng = np.random.default_rng(seed)
    X, y = [], []
    for rec in records:
        if rec["label"] not in keep_labels:
            continue
        sig = rec["signal"]
        if len(sig) < WIN:
            continue
        idx = rng.integers(0, len(sig) - WIN, size=min(n_per, len(sig) // WIN))
        for i in idx:
            X.append(sig[i:i + WIN].astype(np.float32))
            y.append(rec["label"])
    return np.array(X), np.array(y)


def normalize(X):
    """逐窗口 z-score（每个窗口自身均值/标准差），让 AE 聚焦波形形态。"""
    mu, sd = X.mean(1, keepdims=True), X.std(1, keepdims=True) + 1e-6
    return (X - mu) / sd


def recon_error(model, X, bs=64):
    model.eval()
    errs = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            b = torch.from_numpy(X[i:i + bs]).unsqueeze(1).to(DEVICE)
            out = model(b)
            e = ((out - b) ** 2).mean(dim=(1, 2)).cpu().numpy()
            errs.append(e)
    return np.concatenate(errs)


def main():
    set_style()
    records = load_dataset(ROOT / "data" / "CWRU" / "CWRU_data")

    # ---- 1) 只用 Normal 训练 ----
    Xn, _ = make_windows(records, {"Normal"}, n_per=40)
    Xn = normalize(Xn)
    Xn_tr, Xn_val = Xn[:int(0.7 * len(Xn))], Xn[int(0.7 * len(Xn)):]
    print(f"[data] 正常窗口训练={len(Xn_tr)} 验证={len(Xn_val)}（故障样本未进训练）")

    model = AE().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    Xtr_t = torch.from_numpy(Xn_tr).unsqueeze(1).to(DEVICE)
    model.train()
    for epoch in range(40):
        perm = torch.randperm(len(Xtr_t))
        for i in range(0, len(Xtr_t), 64):
            b = Xtr_t[perm[i:i + 64]]
            opt.zero_grad()
            ((model(b) - b) ** 2).mean().backward()
            opt.step()
    print(f"[train] 自编码器训练完成（仅正常数据）")

    # ---- 2) 故障窗口（完全未见）----
    fault_labels = {"Ball", "InnerRace", "OuterRace"}
    Xf, yf = make_windows(records, fault_labels, n_per=40)
    Xf = normalize(Xf)

    # ---- 3) 阈值 = 正常验证误差 95 百分位 ----
    err_val = recon_error(model, Xn_val)
    thr = np.percentile(err_val, 95)
    err_n_te = recon_error(model, Xn_val[:200])
    err_f = recon_error(model, Xf)

    # 标签：0=正常, 1=故障；预测：误差>阈值 判异常
    y_true = np.concatenate([np.zeros(len(err_n_te)), np.ones(len(err_f))])
    y_score = np.concatenate([err_n_te, err_f])
    y_pred = (y_score > thr).astype(int)
    auc = roc_auc_score(y_true, y_score)
    far = (y_pred[:len(err_n_te)] == 1).mean()      # 误报率（正常被判异常）
    dr = (y_pred[len(err_n_te):] == 1).mean()        # 检出率（故障被判异常）
    print(f"[eval]  阈值={thr:.4f}  AUC={auc:.3f}  误报率={far:.3f}  检出率={dr:.3f}")

    # ---- 4) 图 ----
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].hist(err_n_te, bins=30, alpha=0.6, label="正常", color="#1f77b4")
    ax[0].hist(err_f, bins=30, alpha=0.6, label="故障", color="#d62728")
    ax[0].axvline(thr, color="k", ls="--", lw=1.5, label=f"阈值(95%)")
    ax[0].set_xlabel("重构误差 (MSE)")
    ax[0].set_ylabel("窗口数")
    ax[0].set_title("正常 vs 故障 重构误差分布")
    ax[0].legend()

    fpr, tpr, _ = roc_curve(y_true, y_score)
    ax[1].plot(fpr, tpr, color="#2ca02c", lw=2)
    ax[1].plot([0, 1], [0, 1], "k--", lw=1)
    ax[1].set_xlabel("假阳性率（误报）")
    ax[1].set_ylabel("真阳性率（检出）")
    ax[1].set_title(f"ROC (AUC={auc:.3f})")

    # 故障波形重构对比（取一个故障窗口）
    model.eval()
    with torch.no_grad():
        sample = torch.from_numpy(Xf[0:1]).unsqueeze(1).to(DEVICE)
        rec = model(sample).cpu().numpy()[0, 0]
    ax[2].plot(Xf[0], lw=1, color="#1f77b4", label="原始(故障)")
    ax[2].plot(rec, lw=1.2, color="#d62728", label="AE 重构")
    ax[2].set_xlabel("采样点")
    ax[2].set_title("故障窗口：AE 重构不出冲击细节")
    ax[2].legend()
    savefig("anomaly_overview", "engineering")

    print(f"\n[结论] 无需任何故障标签：自编码器靠「正常长什么样」就分出异常 "
          f"(AUC={auc:.2f})。这正是应对「异常样本稀缺」的工业标准做法——"
          f"但注意它只能报「异常」，不能区分「哪种故障」（需后续有标签数据细分）。")


if __name__ == "__main__":
    main()
