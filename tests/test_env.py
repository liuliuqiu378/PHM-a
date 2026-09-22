"""
环境冒烟测试：确认依赖安装正确、GPU 可用、CWRU 数据可读取。

运行：python tests/test_env.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def test_imports():
    import numpy, scipy, matplotlib, sklearn, pandas
    print("[OK] numpy/scipy/matplotlib/sklearn/pandas 可导入")
    try:
        import torch
        print(f"[OK] torch {torch.__version__}")
        if torch.cuda.is_available():
            print(f"[OK] CUDA 可用：{torch.cuda.get_device_name(0)}")
        else:
            print("[WARN] torch 未检测到 GPU，将使用 CPU（仍可用于教程演示）")
    except Exception as e:
        print(f"[FAIL] torch 导入失败：{e}")
        sys.exit(1)


def test_cwru():
    from src.common.cwru_loader import load_dataset, print_summary
    root = ROOT / "data" / "CWRU" / "CWRU_data"
    recs = load_dataset(root)
    print_summary(recs)
    assert len(recs) > 0, "CWRU 数据集为空，请检查 data/CWRU/CWRU_data 路径"


if __name__ == "__main__":
    print("=== 环境冒烟测试 ===")
    test_imports()
    test_cwru()
    print("=== 全部通过 ===")
