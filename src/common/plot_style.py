"""
统一绘图风格与中文字体处理。

本教程所有脚本生成的图都建议通过这里设置 matplotlib 风格，以保证：
- 中文标签正常显示（自动寻找系统中可用的 CJK 字体）
- 图的大小、字号、坐标轴风格一致
- 自动保存到 figures/ 目录
"""

import os
import warnings
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt


# 项目根目录：从 src/common/ 上溯两级
ROOT_DIR = Path(__file__).resolve().parents[2]
FIGURES_DIR = ROOT_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def _pick_chinese_font():
    """尝试从系统字体中找一个支持中文的字体，避免 matplotlib 中文变方块。"""
    candidates = [
        "Noto Sans CJK SC",
        "Noto Sans CJK TC",
        "WenQuanYi Micro Hei",
        "WenQuanYi Zen Hei",
        "SimHei",
        "Microsoft YaHei",
        "Source Han Sans SC",
        "Source Han Serif SC",
    ]
    available = {f.name for f in matplotlib.font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            return name
    # 如果都没找到，用 font_manager 搜索包含 CJK 的字体
    for f in matplotlib.font_manager.fontManager.ttflist:
        if any(k in f.name.lower() for k in ("cjk", "noto", "heiti", "hei", "yahei", "simsun")):
            return f.name
    return None


def set_style():
    """设置 matplotlib 默认绘图风格。"""
    chinese_font = _pick_chinese_font()
    if chinese_font:
        plt.rcParams["font.sans-serif"] = [chinese_font] + plt.rcParams["font.sans-serif"]
        plt.rcParams["axes.unicode_minus"] = False
    else:
        warnings.warn("未找到系统中文字体，matplotlib 中文可能显示为方块。请安装 Noto CJK 或 SimHei。")

    plt.rcParams["figure.figsize"] = (10, 5)
    plt.rcParams["figure.dpi"] = 120
    plt.rcParams["axes.grid"] = True
    plt.rcParams["grid.alpha"] = 0.3
    plt.rcParams["axes.labelsize"] = 12
    plt.rcParams["axes.titlesize"] = 14
    plt.rcParams["legend.fontsize"] = 10
    plt.rcParams["xtick.labelsize"] = 10
    plt.rcParams["ytick.labelsize"] = 10


def savefig(name: str, subdir: str = ""):
    """将当前 figure 保存到 figures/ 目录，返回完整路径。"""
    out_dir = FIGURES_DIR / subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    if not name.endswith(".png"):
        name += ".png"
    out_path = out_dir / name
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"[savefig] {out_path}")
    return out_path


if __name__ == "__main__":
    set_style()
    fig, ax = plt.subplots()
    ax.plot([0, 1, 2], [0, 1, 4], label="示例曲线")
    ax.set_title("中文标题测试")
    ax.set_xlabel("横轴")
    ax.set_ylabel("纵轴")
    ax.legend()
    savefig("style_smoke_test")
    plt.close()
