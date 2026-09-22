"""
模块 05：系统层 / 部署 —— 把「轴承故障诊断」模型从训练态变成可上线服务
=========================================================================

一个 PHM 工程师真正的分水岭，不是把模型在 notebook 里跑出 97%，而是能把模型
**稳定、低延迟、可复现**地部署到边缘设备/工控机里。本脚本演示端到端部署链路：

  训练态 PyTorch 模型
        │  torch.onnx.export
        ▼
  ONNX 中间表示（跨平台、跨框架）
        │  onnxruntime (CPU)
        ▼
  边缘设备上的推理服务（毫秒级、可整合进 PLC/网关）

我们同时给出一张「部署体检卡」：
  - 模型参数量 / 文件体积
  - PyTorch-CPU vs ONNX-Runtime 的单条样本推理延迟
  - 吞吐（条/秒）

这些都是选型时真实要拍板的数字。本脚本自包含地训练一个轻量 1D-CNN 并直接导出。

运行：从 PHM 根目录执行
  python src/module_05_system/deploy_demo.py
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from src.common.plot_style import set_style, savefig
from src.common.cwru_loader import load_dataset, DEFAULT_CLASS_RANGES
from sklearn.model_selection import train_test_split

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FS = 12000.0
WIN = 2048
BATCH_CLASSES = list(DEFAULT_CLASS_RANGES.keys())  # ['Normal','Ball','InnerRace','OuterRace']


class MiniCNN(nn.Module):
    def __init__(self, n_class=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 16, 7, padding=3), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(16, 32, 5, padding=2), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool1d(4),
            nn.Flatten(), nn.Linear(64 * 4, 64), nn.ReLU(), nn.Linear(64, n_class),
        )

    def forward(self, x):
        return self.net(x)


def make_windows(records, n_per=40, seed=0):
    rng = np.random.default_rng(seed)
    X, y = [], []
    for rec in records:
        sig = rec["signal"]
        if len(sig) < WIN:
            continue
        idx = rng.integers(0, len(sig) - WIN, size=min(n_per, len(sig) // WIN))
        for i in idx:
            X.append(sig[i:i + WIN].astype(np.float32))
            y.append(BATCH_CLASSES.index(rec["label"]))
    return np.array(X), np.array(y)


def main():
    set_style()
    print(f"[info] device={DEVICE}")
    records = [r for r in load_dataset(ROOT / "data" / "CWRU" / "CWRU_data")
               if r["label"] in BATCH_CLASSES]
    X, y = make_windows(records)
    print(f"[data] 窗口样本 {X.shape}, 类别 {BATCH_CLASSES}")

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25,
                                          stratify=y, random_state=42)
    model = MiniCNN(len(BATCH_CLASSES)).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()
    Xtr_t = torch.from_numpy(Xtr).unsqueeze(1).to(DEVICE)
    ytr_t = torch.from_numpy(ytr).to(DEVICE)
    # 简短训练：部署演示重点是导出与推理，而非刷精度
    model.train()
    for epoch in range(12):
        perm = torch.randperm(len(Xtr_t))
        for i in range(0, len(Xtr_t), 256):
            b = perm[i:i + 256]
            opt.zero_grad()
            crit(model(Xtr_t[b]), ytr_t[b]).backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        acc = (model(torch.from_numpy(Xte).unsqueeze(1).to(DEVICE)).argmax(1).cpu()
               .numpy() == yte).mean()
    print(f"[train] 验证准确率 {acc:.4f}（轻量模型，仅用于演示部署）")

    # ---- 1) 导出 ONNX ----
    dummy = torch.randn(1, 1, WIN)
    onnx_path = ROOT / "src" / "module_05_system" / "bearing_cnn.onnx"
    torch.onnx.export(model.cpu(), dummy, onnx_path,
                      input_names=["input"], output_names=["output"],
                      dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
                      opset_version=17)
    print(f"[export] ONNX 已导出 -> {onnx_path}")

    # ---- 2) 部署体检卡：参数量 / 体积 / 延迟 ----
    n_params = sum(p.numel() for p in model.parameters())
    pth_size = (ROOT / "src" / "module_05_system" / "_tmp.pt")
    torch.save(model.state_dict(), pth_size)
    pth_bytes = pth_size.stat().st_size
    onnx_bytes = onnx_path.stat().st_size
    pth_size.unlink()

    # PyTorch CPU 延迟
    model.cpu()
    xb = torch.from_numpy(Xte[:256]).unsqueeze(1).float()
    for _ in range(5):
        with torch.no_grad():
            model(xb)
    t0 = time.perf_counter()
    with torch.no_grad():
        for i in range(0, len(xb), 16):
            model(xb[i:i + 16])
    torch_lat = (time.perf_counter() - t0) / len(xb) * 1000  # ms/样本

    # ONNX Runtime 延迟
    import onnxruntime as ort
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    xb_np = xb.numpy()
    for _ in range(5):
        sess.run(None, {"input": xb_np[:16]})
    t0 = time.perf_counter()
    for i in range(0, len(xb_np), 16):
        sess.run(None, {"input": xb_np[i:i + 16]})
    ort_lat = (time.perf_counter() - t0) / len(xb_np) * 1000

    print("\n========== 部署体检卡 ==========")
    print(f"参数量            : {n_params:,}")
    print(f"PyTorch 权重体积  : {pth_bytes/1024:.1f} KB")
    print(f"ONNX 模型体积     : {onnx_bytes/1024:.1f} KB")
    print(f"PyTorch-CPU 延迟  : {torch_lat:.3f} ms/样本  ({1000/torch_lat:.0f} 条/秒)")
    print(f"ONNX-RT 延迟      : {ort_lat:.3f} ms/样本  ({1000/ort_lat:.0f} 条/秒)")
    print("===============================")

    # 可视化体检卡
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(["PyTorch", "ONNX-RT"], [torch_lat, ort_lat], color=["#1f77b4", "#ff7f0e"])
    for i, v in enumerate([torch_lat, ort_lat]):
        axes[0].text(i, v + 0.02, f"{v:.2f}ms", ha="center")
    axes[0].set_title("单样本推理延迟 (CPU)")
    axes[0].set_ylabel("ms / 样本")
    axes[1].bar(["PyTorch", "ONNX"], [pth_bytes / 1024, onnx_bytes / 1024],
                color=["#1f77b4", "#ff7f0e"])
    for i, v in enumerate([pth_bytes / 1024, onnx_bytes / 1024]):
        axes[1].text(i, v + 1, f"{v:.1f}KB", ha="center")
    axes[1].set_title("模型文件体积")
    axes[1].set_ylabel("KB")
    savefig("deploy_card", "system")

    # 推理一致性校验：ONNX 与 PyTorch 输出应几乎一致
    with torch.no_grad():
        pt_out = model(xb[:8]).numpy()
    ort_out = sess.run(None, {"input": xb_np[:8]})[0]
    max_diff = float(np.abs(pt_out - ort_out).max())
    print(f"[check] ONNX 与 PyTorch 输出最大差异 = {max_diff:.2e}（应接近 0）")
    assert max_diff < 1e-3, "导出不一致！"


if __name__ == "__main__":
    main()
