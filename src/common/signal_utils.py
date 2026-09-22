"""
信号处理公共工具：FFT、包络解调、时域/频域特征、轴承故障特征频率。

PHM 里 90% 的"看图诊断"都建立在这些函数之上。
"""

import numpy as np
from scipy.signal import butter, filtfilt, hilbert, welch


def fft_spectrum(x: np.ndarray, fs: float):
    """返回 (freq, mag) 单边幅度谱（线性幅值，已除 N）。"""
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    freq = np.fft.rfftfreq(n, d=1.0 / fs)
    mag = np.abs(np.fft.rfft(x - np.mean(x))) / n
    return freq, mag


def bandpass(x: np.ndarray, fs: float, low: float, high: float, order: int = 4):
    """零相位带通滤波。"""
    b, a = butter(order, [low / (fs / 2), high / (fs / 2)], btype="band")
    return filtfilt(b, a, x)


def envelope_spectrum(x: np.ndarray, fs: float, band=(2000.0, 5000.0)):
    """
    包络解调：带通 -> 希尔伯特包络 -> 对包络做 FFT。
    轴承点蚀会产生周期性冲击，冲击被机器结构共振放大，包络谱上能看到故障特征频率。
    返回 (freq, env_mag)。
    """
    x_bp = bandpass(x - np.mean(x), fs, band[0], band[1])
    env = np.abs(hilbert(x_bp))
    freq = np.fft.rfftfreq(len(env), d=1.0 / fs)
    env_mag = np.abs(np.fft.rfft(env - np.mean(env))) / len(env)
    return freq, env_mag


def time_features(x: np.ndarray) -> dict:
    """常用时域特征。"""
    x = np.asarray(x, dtype=np.float64)
    x = x - np.mean(x)
    rms = np.sqrt(np.mean(x**2))
    peak = np.max(np.abs(x))
    std = np.std(x)
    kurt = np.mean(x**4) / (rms**4 + 1e-12)
    crest = peak / (rms + 1e-12)
    return {
        "rms": float(rms),
        "peak": float(peak),
        "std": float(std),
        "kurtosis": float(kurt),
        "crest_factor": float(crest),
        "pp": float(np.ptp(x)),
    }


def fault_frequencies_6205(rpm: float) -> dict:
    """
    CWRU 使用的 6205-2RS JEM 轴承几何参数计算的故障特征频率。
    几何：Z=9 滚珠, 滚珠径 d=7.94mm, 节圆径 D=39.04mm, 接触角 phi=0。
    返回以 Hz 为单位的 BPFO/BPFI/BSF/FTF。
    """
    Z, d, D, phi = 9, 7.94, 39.04, 0.0
    f_shaft = rpm / 60.0
    c = np.cos(phi)
    bpfo = Z / 2 * (1 - d / D * c) * f_shaft
    bpfi = Z / 2 * (1 + d / D * c) * f_shaft
    bsf = D / (2 * d) * (1 - (d / D * c) ** 2) * f_shaft
    ftf = 1 / 2 * (1 - d / D * c) * f_shaft
    return {"BPFO": bpfo, "BPFI": bpfi, "BSF": bsf, "FTF": ftf}


def feature_vector(x: np.ndarray, fs: float, rpm: float = None) -> np.ndarray:
    """
    组合特征：时域(5) + 频带能量占比(5) + 包络谱在故障特征频率处的幅值(4)。
    共 14 维，物理可解释，适合作为传统 ML 的输入。
    """
    tf = time_features(x)
    freq, mag = fft_spectrum(x, fs)
    bands = [(0, 100), (100, 500), (500, 1000), (1000, 2000), (2000, 5000)]
    total = (mag ** 2).sum() + 1e-12
    band_feats = []
    for lo, hi in bands:
        mask = (freq >= lo) & (freq < hi)
        band_feats.append(float((mag[mask] ** 2).sum() / total))

    feats = [tf["rms"], tf["peak"], tf["kurtosis"], tf["crest_factor"], tf["pp"]] + band_feats

    if rpm:
        ff = fault_frequencies_6205(rpm)
        ef, em = envelope_spectrum(x, fs)
        for fv in ff.values():
            idx = int(np.argmin(np.abs(ef - fv)))
            feats.append(float(em[idx]))
    else:
        feats += [0.0, 0.0, 0.0, 0.0]
    return np.array(feats, dtype=np.float64)


if __name__ == "__main__":
    print("6205 轴承故障特征频率（rpm=1797）：")
    for k, v in fault_frequencies_6205(1797.0).items():
        print(f"  {k}: {v:.1f} Hz")
