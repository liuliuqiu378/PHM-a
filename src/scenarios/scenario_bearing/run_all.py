"""
场景 A：电机轴承故障诊断 —— 一键复现。

把第 1~3 层串成一个完整项目：CWRU 数据 -> 信号处理 -> ML + 1D-CNN 诊断 -> 出结果卡片。

运行：python src/scenarios/scenario_bearing/run_all.py
依赖：src.module_03_diagnosis.train_ml / train_1dcnn（脚本已有 __main__ 入口）
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))


def main():
    print("=" * 60)
    print("场景 A：电机轴承故障诊断（CWRU 数据集）")
    print("=" * 60)
    print("[1/3] 传统 ML 诊断（RF/SVM/KNN）...")
    from src.module_03_diagnosis import train_ml
    train_ml.main()

    print("\n[2/3] 1D-CNN 端到端诊断（原始波形）...")
    from src.module_03_diagnosis import train_1dcnn
    train_1dcnn.main()

    print("\n[3/3] 结果卡片")
    print("  - 数据：CWRU 12k 驱动端，4 类（Normal/Ball/InnerRace/OuterRace）")
    print("  - 特征：时域 + 频带能量 + 包络故障频率（14 维）")
    print("  - 输出图：figures/diagnosis/{ml,cnn}_confusion_matrix.png")
    print("  - 结论：传统 ML ~97%+，1D-CNN ~99%+（详见 docs/03_diagnosis.md）")
    print("=" * 60)


if __name__ == "__main__":
    main()
