"""
第 3 层 - 故障诊断（1D-CNN 端到端，吃原始波形）。

提供两种评测协议，便于讲清"同工况"与"跨工况"的区别：
- 窗口级划分（训练/测试窗口来自同一批文件）：对标文献，~99%，但存在同文件泄漏；
- 文件级划分（训练/测试来自不同文件，可能不同负载）：更贴近现场，会暴露跨工况泛化差距。

运行：python src/module_03_diagnosis/train_1dcnn.py
产出：figures/diagnosis/cnn_confusion_matrix.png（窗口级）
"""

import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.common.plot_style import set_style, savefig
from src.common.cwru_loader import load_dataset

import matplotlib.pyplot as plt

CWRU_DIR = ROOT / "data" / "CWRU" / "CWRU_data"
FS = 12000.0
SEED = 42
WIN = 2048
STRIDE = 1024
MAX_WIN = 80
EPOCHS = 60
BATCH = 128

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def make_windows(records):
    """返回 [(window_array, label), ...]，已归一化。"""
    out = []
    for rec in records:
        x = rec["signal"]
        n = len(x)
        cnt = 0
        for s in range(0, n - WIN, STRIDE):
            w = x[s:s + WIN].astype(np.float32)
            w = (w - w.mean()) / (w.std() + 1e-8)
            out.append((w, rec["label"]))
            cnt += 1
            if cnt >= MAX_WIN:
                break
    return out


class WinDataset(Dataset):
    def __init__(self, items, classes):
        self.X = np.array([it[0] for it in items], dtype=np.float32)
        self.y = np.array([classes.index(it[1]) for it in items], dtype=np.int64)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return torch.from_numpy(self.X[i]).unsqueeze(0), self.y[i]


class CNN1D(nn.Module):
    def __init__(self, n_class):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 16, 15, padding=7), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(16, 32, 15, padding=7), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 15, padding=7), nn.ReLU(), nn.MaxPool1d(2),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
        )
        self.clf = nn.Sequential(nn.Dropout(0.5), nn.Linear(64, n_class))

    def forward(self, x):
        return self.clf(self.features(x))


def train_eval(train_items, test_items, classes):
    n_class = len(classes)
    train_loader = DataLoader(WinDataset(train_items, classes), batch_size=BATCH, shuffle=True, num_workers=2)
    test_loader = DataLoader(WinDataset(test_items, classes), batch_size=BATCH, shuffle=False, num_workers=2)
    model = CNN1D(n_class).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    for epoch in range(1, EPOCHS + 1):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            criterion(model(xb), yb).backward()
            optimizer.step()
        scheduler.step()
    model.eval()
    yt, yp = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            yp.extend(model(xb.to(DEVICE)).argmax(1).cpu().numpy())
            yt.extend(yb.numpy())
    return accuracy_score(yt, yp), yt, yp


def main():
    set_style()
    print(f"device = {DEVICE}")
    records = load_dataset(CWRU_DIR)
    classes = sorted(set(r["label"] for r in records))
    print(f"类别：{classes}")

    # 协议 A：窗口级划分（同工况，对标文献）
    all_items = make_windows(records)
    tr, te = train_test_split(all_items, test_size=0.25, stratify=[i[1] for i in all_items], random_state=SEED)
    acc_win, yt_win, yp_win = train_eval(tr, te, classes)
    print(f"\n[窗口级划分] 1D-CNN 准确率：{acc_win:.4f}")
    print(classification_report(yt_win, yp_win, target_names=classes, digits=4))

    # 协议 B：文件级划分（跨工况，更贴近现场）
    tr_rec, te_rec = train_test_split(records, test_size=0.25, stratify=[r["label"] for r in records], random_state=SEED)
    tr_items = make_windows(tr_rec)
    te_items = make_windows(te_rec)
    acc_file, yt_file, yp_file = train_eval(tr_items, te_items, classes)
    print(f"\n[文件级划分] 1D-CNN 准确率：{acc_file:.4f}（训练文件 {len(tr_rec)}，测试文件 {len(te_rec)}）")

    # 保存窗口级混淆矩阵
    cm = confusion_matrix(yt_win, yp_win, labels=list(range(len(classes))))
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes))); ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right"); ax.set_yticklabels(classes)
    ax.set_xlabel("预测标签"); ax.set_ylabel("真实标签")
    ax.set_title(f"1D-CNN 混淆矩阵 (窗口级 acc={acc_win:.3f})")
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="red" if i == j else "black", fontsize=11)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    savefig("cnn_confusion_matrix", subdir="diagnosis")
    plt.close()

    print(f"\n[结论] 同工况 {acc_win:.3f} vs 跨工况 {acc_file:.3f}："
          f"差距说明原始波形对负载/工况敏感，物理特征或域适应可缓解（见 docs/03_diagnosis.md）。")


if __name__ == "__main__":
    main()
