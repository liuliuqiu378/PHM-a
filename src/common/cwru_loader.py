"""
CWRU 轴承数据集加载器。

本地 CWRU 数据的 .mat 文件里通常包含形如 X{id}_DE_time 的变量。
这里按变量 ID 把信号读出来，并按标准 CWRU 编号规则映射到 4 类故障标签：

- Normal     : 97, 98, 99, 100
- Ball       : 105-112(B007), 169-177(B014), 209-215(B021)
- InnerRace  : 118-125(IR007), 185-192(IR014), 222-229(IR021)
- OuterRace  : 130-138(OR007), 197-204(OR014), 234-241(OR021)

说明：161 个文件里还包含 FE 端、48k 采样、基座等数据；本教程默认只用
12k 驱动端（DE）的清晰分类子集。后续场景可扩展为 10 类或引入多传感器。
"""

import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from scipy.io import loadmat


DEFAULT_CLASS_RANGES = {
    "Normal": [(97, 100)],
    "Ball": [(105, 112), (169, 177), (209, 215)],
    "InnerRace": [(118, 125), (185, 192), (222, 229)],
    "OuterRace": [(130, 138), (197, 204), (234, 241)],
}


def build_default_label_map(class_ranges: Dict[str, List[tuple]] = None) -> Dict[int, str]:
    """把区间映射展开成 id -> label 的字典。"""
    if class_ranges is None:
        class_ranges = DEFAULT_CLASS_RANGES
    label_map = {}
    for label, ranges in class_ranges.items():
        for start, end in ranges:
            for vid in range(start, end + 1):
                label_map[vid] = label
    return label_map


DEFAULT_LABEL_MAP = build_default_label_map()


def class_of_id(var_id: int, label_map: Optional[Dict[int, str]] = None) -> Optional[str]:
    if label_map is None:
        label_map = DEFAULT_LABEL_MAP
    return label_map.get(var_id)


def load_mat(path: Path) -> Dict[int, dict]:
    """读取单个 .mat，返回 {var_id: {signal, rpm, key, file}}（仅驱动端 DE 信号）。"""
    path = Path(path)
    try:
        mat = loadmat(str(path))
    except Exception:
        # scipy 读不了的（如 MATLAB v7.3 / HDF5 格式）回退到 h5py
        return load_mat_h5(path)
    samples = {}
    rpm = None
    m_file = re.search(r"(\d+)", path.stem)
    rpm_key = f"X{m_file.group(1)}RPM" if m_file else None
    if rpm_key and rpm_key in mat:
        rpm = int(mat[rpm_key][0, 0])

    for key in mat:
        if key.startswith("__"):
            continue
        m = re.match(r"X(\d+)_DE_time", key)
        if m:
            vid = int(m.group(1))
            signal = mat[key].ravel().astype(np.float64)
            samples[vid] = {"signal": signal, "rpm": rpm, "key": key, "file": path.name}
    return samples


def load_mat_h5(path: Path) -> Dict[int, dict]:
    """用 h5py 读取 MATLAB v7.3 (HDF5) 格式的 .mat，返回与 load_mat 相同结构。"""
    import h5py
    path = Path(path)
    samples = {}
    with h5py.File(str(path), "r") as f:
        keys = list(f.keys())
        m_file = re.search(r"(\d+)", path.stem)
        rpm_key = f"X{m_file.group(1)}RPM" if m_file else None
        rpm = None
        if rpm_key and rpm_key in f:
            rpm = int(np.array(f[rpm_key]).ravel()[0])
        for key in keys:
            m = re.match(r"X(\d+)_DE_time", key)
            if m:
                vid = int(m.group(1))
                signal = np.array(f[key]).ravel().astype(np.float64)
                samples[vid] = {"signal": signal, "rpm": rpm, "key": key, "file": path.name}
    return samples


def load_dataset(
    root_dir: Path,
    label_map: Optional[Dict[int, str]] = None,
    keep_unlabeled: bool = False,
) -> List[dict]:
    """批量读取 CWRU .mat，返回带标签的样本列表。"""
    root_dir = Path(root_dir)
    label_map = label_map or DEFAULT_LABEL_MAP
    records = []
    skipped = []
    for mat_path in sorted(root_dir.glob("*.mat")):
        try:
            samples = load_mat(mat_path)
        except Exception as e:
            skipped.append((mat_path.name, str(e)))
            continue
        for vid, info in samples.items():
            label = class_of_id(vid, label_map)
            if label is None and not keep_unlabeled:
                continue
            records.append(
                {
                    "file": info["file"],
                    "var_id": vid,
                    "signal": info["signal"],
                    "rpm": info["rpm"],
                    "label": label,
                }
            )
    if skipped:
        print(f"[warn] 跳过 {len(skipped)} 个无法读取的 .mat：{skipped[:5]}")
    return records


def class_distribution(records: List[dict]) -> Dict[str, int]:
    dist = {}
    for r in records:
        dist[r["label"]] = dist.get(r["label"], 0) + 1
    return dist


def print_summary(records: List[dict]):
    print(f"加载样本数：{len(records)}")
    print("类别分布：")
    for label, count in sorted(class_distribution(records).items()):
        print(f"  {label:12s}: {count}")
    if records:
        r = records[0]
        print(f"\n示例：file={r['file']}, var_id={r['var_id']}, label={r['label']}, "
              f"rpm={r['rpm']}, len(signal)={len(r['signal'])}")


if __name__ == "__main__":
    import os
    root = Path(__file__).resolve().parents[2] / "data" / "CWRU" / "CWRU_data"
    recs = load_dataset(root)
    print_summary(recs)
