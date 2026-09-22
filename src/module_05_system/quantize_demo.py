"""
模块 05（续）：INT8 量化 —— 把 ONNX 模型再压一轮、跑得更快
=========================================================================

训练完、导出 ONNX 之后，工业部署里还差关键一步：**量化**。

本脚本复用 `deploy_demo.py` 的 MiniCNN 模型定义，自包含地演示 INT8 量化，
并对比三项真实指标：模型体积 / 单样本延迟 / 测试集精度。

两个重要工程事实（都用代码验证，而非口说）：
  1) 动态量化（Dynamic Quantization）对卷积层会生成 ConvInteger 节点，
     但当前 onnxruntime 的 CPU 执行提供器没有该算子实现 —— 模型能导出、
     体积也能压下去，却**跑不起来**。脚本会如实捕获这个限制。
  2) 工业上真正能部署的 INT8 卷积，走的是**静态量化（Static Quantization）**：
     用一小批校准数据确定激活的量化范围，生成 QLinearConv，CPU EP 原生支持。
     本脚本以静态量化得到可运行、体积 ≈1/3、精度几乎无损的 INT8 模型。

运行（从 PHM 根目录）：
  python src/module_05_system/quantize_demo.py
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
from src.module_05_system.deploy_demo import MiniCNN, make_windows, BATCH_CLASSES, WIN

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def train_model():
    records = [r for r in load_dataset(ROOT / "data" / "CWRU" / "CWRU_data")
               if r["label"] in BATCH_CLASSES]
    X, y = make_windows(records)
    from sklearn.model_selection import train_test_split
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25,
                                          stratify=y, random_state=42)
    model = MiniCNN(len(BATCH_CLASSES)).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()
    Xtr_t = torch.from_numpy(Xtr).unsqueeze(1).to(DEVICE)
    ytr_t = torch.from_numpy(ytr).to(DEVICE)
    model.train()
    for _ in range(12):
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
    print(f"[train] 验证准确率 {acc:.4f}")
    return model.cpu(), Xte, yte


class _CalibReader:
    """静态量化所需的校准数据读取器：提供若干输入 batch。"""
    def __init__(self, data, n=64):
        self.buf = [{"input": data[i].astype(np.float32).reshape(1, 1, WIN)}
                    for i in range(min(n, len(data)))]
        self.i = 0

    def get_next(self):
        if self.i >= len(self.buf):
            return None
        b = self.buf[self.i]
        self.i += 1
        return b


def benchmark(sess, xb_np, repeat=20):
    for _ in range(5):
        sess.run(None, {"input": xb_np[:16]})
    t0 = time.perf_counter()
    for _ in range(repeat):
        for i in range(0, len(xb_np), 16):
            sess.run(None, {"input": xb_np[i:i + 16]})
    lat = (time.perf_counter() - t0) / repeat / len(xb_np) * 1000
    return lat, 1000 / lat


def main():
    set_style()
    model, Xte, yte = train_model()
    xb_np = Xte[:256].astype(np.float32).reshape(-1, 1, WIN)

    # ---- 1) 导出 FP32 ONNX ----
    fp32_path = ROOT / "src" / "module_05_system" / "bearing_cnn.onnx"
    torch.onnx.export(model, torch.randn(1, 1, WIN), fp32_path,
                      input_names=["input"], output_names=["output"],
                      dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
                      opset_version=17)
    import onnxruntime as ort
    from onnxruntime.quantization import (quantize_dynamic, quantize_static,
                                          CalibrationDataReader, QuantType, QuantFormat)

    fp32_kb = fp32_path.stat().st_size / 1024
    print(f"[size]  FP32 = {fp32_kb:.1f} KB")

    # ---- 2) 动态量化：演示其「卷积局限」----
    dyn_path = ROOT / "src" / "module_05_system" / "bearing_cnn.dyn.onnx"
    try:
        quantize_dynamic(str(fp32_path), str(dyn_path), weight_type=QuantType.QInt8)
        dyn_kb = dyn_path.stat().st_size / 1024
        print(f"[dyn ]  动态量化体积 = {dyn_kb:.1f} KB（权重已压成 INT8）")
        try:
            ort.InferenceSession(str(dyn_path), providers=["CPUExecutionProvider"])
            dyn_runnable = True
        except Exception as e:
            dyn_runnable = False
            print(f"[dyn ]  ⚠ 当前 CPU EP 无法推理动态量化模型：{type(e).__name__}")
            print(f"[dyn ]  原因：动态量化在卷积层生成 ConvInteger，本构建未实现该算子。")
    except Exception as e:
        print(f"[dyn ] 动态量化失败：{e}")
        dyn_kb, dyn_runnable = float("nan"), False

    # ---- 3) 静态量化：得到可运行的 INT8 模型 ----
    sta_path = ROOT / "src" / "module_05_system" / "bearing_cnn.int8.onnx"
    quantize_static(str(fp32_path), str(sta_path),
                    calibration_data_reader=_CalibReader(Xte),
                    quant_format=QuantFormat.QOperator, weight_type=QuantType.QInt8)
    sta_kb = sta_path.stat().st_size / 1024
    print(f"[stat]  静态量化体积 = {sta_kb:.1f} KB（QLinearConv，CPU EP 原生支持）")

    # ---- 4) 延迟 / 精度对比（FP32 vs 静态 INT8）----
    fp32_sess = ort.InferenceSession(str(fp32_path), providers=["CPUExecutionProvider"])
    sta_sess = ort.InferenceSession(str(sta_path), providers=["CPUExecutionProvider"])
    fp32_lat, fp32_thr = benchmark(fp32_sess, xb_np)
    sta_lat, sta_thr = benchmark(sta_sess, xb_np)

    def acc_of(sess):
        pred = []
        for i in range(0, len(xb_np), 16):
            pred.append(sess.run(None, {"input": xb_np[i:i + 16]})[0].argmax(1))
        return (np.concatenate(pred) == yte[:len(xb_np)]).mean()
    fp32_acc, sta_acc = acc_of(fp32_sess), acc_of(sta_sess)

    fp32_out = fp32_sess.run(None, {"input": xb_np[:8]})[0]
    sta_out = sta_sess.run(None, {"input": xb_np[:8]})[0]
    max_diff = float(np.abs(fp32_out - sta_out).max())

    print(f"[lat]   FP32={fp32_lat:.3f}ms  INT8={sta_lat:.3f}ms "
          f"({'更快' if sta_lat < fp32_lat else '相近'})")
    print(f"[thr]   FP32={fp32_thr:.0f}条/秒  INT8={sta_thr:.0f}条/秒")
    print(f"[acc]   FP32={fp32_acc:.4f}  INT8={sta_acc:.4f}  "
          f"损失={fp32_acc - sta_acc:+.4f}")
    print(f"[diff]  量化 logits 最大差异 = {max_diff:.2e}（应远小于决策边界）")

    # ---- 5) 可视化对比卡 ----
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    labels = ["FP32", "动态(不可跑)", "静态INT8"] if not dyn_runnable else ["FP32", "动态", "静态"]
    sizes = [fp32_kb, (dyn_kb if not np.isnan(dyn_kb) else 0), sta_kb]
    colors = ["#1f77b4", "#d62728", "#2ca02c"]
    axes[0].bar(labels, sizes, color=colors)
    for i, v in enumerate(sizes):
        axes[0].text(i, v + 1, f"{v:.1f}KB", ha="center")
    axes[0].set_title("模型体积")
    axes[0].set_ylabel("KB")

    axes[1].bar(["FP32", "INT8"], [fp32_lat, sta_lat], color=["#1f77b4", "#2ca02c"])
    for i, v in enumerate([fp32_lat, sta_lat]):
        axes[1].text(i, v + 0.002, f"{v:.3f}ms", ha="center")
    axes[1].set_title("单样本推理延迟 (CPU)")
    axes[1].set_ylabel("ms / 样本")

    axes[2].bar(["FP32", "INT8"], [fp32_acc, sta_acc], color=["#1f77b4", "#2ca02c"])
    for i, v in enumerate([fp32_acc, sta_acc]):
        axes[2].text(i, v + 0.005, f"{v:.4f}", ha="center")
    axes[2].set_title("测试集准确率")
    axes[2].set_ylim(0, 1.05)
    savefig("quant_card", "system")

    print(f"\n[结论] 静态 INT8：体积 {fp32_kb / sta_kb:.1f}x 更小、"
          f"延迟{'更优' if sta_lat < fp32_lat else '相近'}、"
          f"精度损失 {fp32_acc - sta_acc:+.4f}（几乎无损）。"
          f"动态量化虽也能压体积，但卷积节点在当前 CPU EP 跑不起来 —— "
          f"工业部署应优先静态量化或选用支持 INT8 卷积的 EP（QNN/DML/TensorRT）。")


if __name__ == "__main__":
    main()
