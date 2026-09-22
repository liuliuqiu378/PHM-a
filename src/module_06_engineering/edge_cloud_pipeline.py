"""
模块 06：工程实战（二）—— 边缘-云数据管线：传感器 → 边缘提特征 → MQTT 只传特征/告警 → 云聚合
================================================================================================

你原话：「模型部署在哪（端侧/云侧）、数据怎么传、传什么」。本脚本把这一整条链路
做成**可运行**的 demo，回答三个工程问题：

  1) 传什么？  边缘不传原始波形（巨大），只传「特征 + 健康分 + 告警」（几十~几百字节）。
  2) 怎么传？  用 MQTT 风格的发布/订阅（topic 分层、pub/sub 解耦），本脚本用本地 broker
              仿真，生产环境把 `LocalMQTTBroker` 换成 paho-mqtt + Mosquitto/EMQX 即可。
  3) 断了怎么办？ 边缘有「断网缓存」：broker 不可达时本地缓冲，恢复后续传，关键片段不丢。

真实链路：
  传感器(DAQ) → 边缘网关(滤波+提特征+初筛告警) → MQTT → 云平台(聚合+趋势+生成工单)

本脚本用合成振动流演示（健康 → 渐变退化 → 持续故障），并实测带宽节省比。
运行（从 PHM 根目录）：
  python src/module_06_engineering/edge_cloud_pipeline.py
"""
import sys
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib.pyplot as plt

from src.common.plot_style import set_style, savefig

WIN = 512          # 每窗口采样点数（生产常用 1024~4096；demo 取小值跑得快）
FS = 12000.0       # 采样率 Hz
N_WIN = 240        # 模拟 240 个窗口（约 20s 流）
PLANT = "plantA"
DEVICE = "pump-01"
THROTTLE_FULL = 8  # 每 8 个窗口才传一次「完整特征」，其余只传轻量心跳（省带宽）


# ------------------------- 1. 传感器仿真 -------------------------
def sensor_stream(n_win, win, fs, seed=0):
    """合成振动流：前段健康，第 120 窗起注入周期冲击(轴承故障)并渐变加重。"""
    rng = np.random.default_rng(seed)
    # 健康成分：转频 30Hz + 谐波 + 高斯噪声
    t_all = np.arange(n_win * win) / fs
    base = (0.6 * np.sin(2 * np.pi * 30 * t_all)
            + 0.3 * np.sin(2 * np.pi * 60 * t_all)
            + 0.15 * np.sin(2 * np.pi * 90 * t_all))
    base += rng.normal(0, 0.08, size=t_all.size)
    # 故障冲击：BPFI≈162Hz 的周期性尖脉冲，振幅随窗口线性增长
    fault = np.zeros_like(t_all)
    for w in range(n_win):
        if w < 120:
            continue
        ramp = (w - 120) / (n_win - 120)          # 0 → 1 渐变
        amp = 0.2 + 1.6 * ramp
        seg = np.arange(win)
        imp = seg % int(fs / 162) == 0            # 每 ~74 点一个冲击
        fault[w * win:(w + 1) * win] += amp * imp * rng.normal(1, 0.2, size=win)
    sig = base + fault
    return sig


# ------------------------- 2. 边缘：特征提取 -------------------------
def extract_features(x):
    """从一窗口原始振动提 12 维特征（这就是「传什么」的核心内容）。"""
    x = x - x.mean()
    rms = np.sqrt(np.mean(x ** 2))
    peak = np.max(np.abs(x))
    std = x.std() + 1e-9
    kurt = np.mean((x / std) ** 4)
    skew = np.mean((x / std) ** 3)
    crest = peak / (rms + 1e-9)
    shape = rms / (np.mean(np.abs(x)) + 1e-9)
    impulse = peak / (np.mean(np.abs(x)) + 1e-9)
    # 频域
    sp = np.abs(np.fft.rfft(x)) ** 2
    freqs = np.fft.rfftfreq(len(x), 1 / FS)
    sp_sum = sp.sum() + 1e-9
    dom_f = freqs[np.argmax(sp)]
    centroid = float(np.sum(freqs * sp) / sp_sum)
    hf = np.sum(sp[freqs > 300]) / sp_sum           # 高频能量占比
    return {
        "rms": float(rms), "peak": float(peak), "kurtosis": float(kurt),
        "skewness": float(skew), "crest": float(crest), "shape": float(shape),
        "impulse": float(impulse), "dom_freq": float(dom_f),
        "spectral_centroid": centroid, "hf_ratio": float(hf),
        "mean": float(x.mean()), "std": float(x.std()),
    }


# ------------------------- 3. MQTT 风格本地 broker（仿真） -------------------------
class LocalMQTTBroker:
    """极简 pub/sub，支持 `+`(单层) / `#`(多层) 通配符。生产换 paho-mqtt 即可。

    online 开关模拟工厂网络抖动：offline 时 publish 返回 False，边缘据此缓存。
    """
    def __init__(self):
        self.subs = {}        # topic_pattern -> [callback]
        self.online = True

    def subscribe(self, pattern, cb):
        self.subs.setdefault(pattern, []).append(cb)

    def _match(self, topic, pattern):
        t, p = topic.split("/"), pattern.split("/")
        for tp, pp in zip(t, p):
            if pp == "#":
                return True
            if pp == "+":
                continue
            if tp != pp:
                return False
        return len(t) == len(p)

    def publish(self, topic, payload_bytes, qos=0):
        if not self.online:
            return False                       # 网络断了 → 边缘去缓存
        for pat, cbs in self.subs.items():
            if self._match(topic, pat):
                for cb in cbs:
                    cb(topic, payload_bytes)
        return True


# ------------------------- 4. 边缘节点 -------------------------
class EdgeNode:
    def __init__(self, broker, plant, device, win):
        self.broker = broker
        self.plant, self.device, self.win = plant, device, win
        self.seq = 0
        self.base_kurt = []        # 用前段健康数据建「正常基线」
        self.base_rms = []
        self.kurt_thr = None
        self.rms_thr = None
        self.buffer = []           # 断网缓存
        self.sent_features = []     # 记录实际传出的字节（带宽统计）
        self.sent_raw_equiv = 0     # 若传原始波形会是多少字节

    def _alert(self, feats):
        """用基线 z-score 判告警（>3σ）。无需任何故障标签。"""
        if self.kurt_thr is None:
            return False, 0.0
        z_k = (feats["kurtosis"] - self.kurt_thr[0]) / (self.kurt_thr[1] + 1e-6)
        z_r = (feats["rms"] - self.rms_thr[0]) / (self.rms_thr[1] + 1e-6)
        score = float(max(z_k, z_r))
        return score > 3.0, score

    def process(self, raw):
        feats = extract_features(raw)
        self.sent_raw_equiv += raw.nbytes            # 原始波形字节（对照用）

        # 建基线（前 60 窗视为健康）
        if self.seq < 60:
            self.base_kurt.append(feats["kurtosis"])
            self.base_rms.append(feats["rms"])
            if self.seq == 59:
                self.kurt_thr = (np.mean(self.base_kurt), np.std(self.base_kurt))
                self.rms_thr = (np.mean(self.base_rms), np.std(self.base_rms))
                print(f"[edge] 健康基线建好  kurtosis≈{self.kurt_thr[0]:.2f}±{self.kurt_thr[1]:.2f}"
                      f"  rms≈{self.rms_thr[0]:.3f}")

        alert, score = self._alert(feats)

        # 心跳 payload（轻量，每窗都传）：只传 3 个数
        heartbeat = {
            "device": self.device, "seq": self.seq,
            "ts": time.time(), "rms": round(feats["rms"], 4),
            "health_score": round(score, 3), "alert": alert,
        }
        # 完整特征 payload（重，按 THROTTLE_FULL 节流，或告警时必传）
        full = {**heartbeat, "features": {k: round(v, 4) for k, v in feats.items()}}
        send_full = (self.seq % THROTTLE_FULL == 0) or alert
        payload = full if send_full else heartbeat

        topic = f"{self.plant}/{self.device}/telemetry"
        if alert:
            topic = f"{self.plant}/{self.device}/alert"
        pb = json.dumps(payload).encode("utf-8")

        # 发布；失败则进断网缓存
        if self.broker.publish(topic, pb):
            self.sent_features.append(len(pb))
        else:
            self.buffer.append(pb)
        self.seq += 1
        return feats, alert, score, send_full


# ------------------------- 5. 云平台：聚合 + 工单 -------------------------
class CloudAggregator:
    def __init__(self, broker, plant, device):
        self.plant, self.device = plant, device
        self.recv_bytes = 0
        self.rms_hist = []
        self.alert_seqs = []
        self.work_orders = []
        self.open_wo = None          # 当前打开的工单 (wo, open_seq, score)
        self.since_alert = 999       # 距上次告警的窗口数（迟滞用）
        self.cooldown = 12           # 连续 12 窗无告警才关闭工单（吸收抖动）
        broker.subscribe(f"{plant}/+/telemetry", self.on_msg)
        broker.subscribe(f"{plant}/+/alert", self.on_msg)

    def on_msg(self, topic, payload_bytes):
        self.recv_bytes += len(payload_bytes)
        msg = json.loads(payload_bytes.decode("utf-8"))
        self.rms_hist.append(msg["rms"])
        alert = bool(msg.get("alert"))
        if alert:
            self.alert_seqs.append(msg["seq"])
            self.since_alert = 0
            if self.open_wo is None:        # 上升沿：开新工单
                wo = f"WO-{len(self.work_orders)+1:03d}"
                self.open_wo = (wo, msg["seq"], round(msg["health_score"], 2))
                self.work_orders.append(self.open_wo)
                print(f"[cloud] 开新工单 {wo} @seq={msg['seq']}  健康分={msg['health_score']:.2f}（持续告警收敛为单次事件）")
        else:
            self.since_alert += 1
            if self.open_wo is not None and self.since_alert >= self.cooldown:
                print(f"[cloud] 关闭工单 {self.open_wo[0]} @seq={msg['seq']}（连续 {self.cooldown} 窗无告警）")
                self.open_wo = None


# ------------------------- 6. 主流程 -------------------------
def main():
    set_style()
    rng_seed = 0
    sig = sensor_stream(N_WIN, WIN, FS, seed=rng_seed)
    broker = LocalMQTTBroker()
    edge = EdgeNode(broker, PLANT, DEVICE, WIN)
    cloud = CloudAggregator(broker, PLANT, DEVICE)

    kurt_hist, score_hist, alert_hist = [], [], []
    print(f"[sim] 模拟 {N_WIN} 窗口振动流（健康 → 渐变退化 → 持续故障），FS={FS:.0f}Hz")
    for w in range(N_WIN):
        raw = sig[w * WIN:(w + 1) * WIN]
        feats, alert, score, _ = edge.process(raw)
        kurt_hist.append(feats["kurtosis"])
        score_hist.append(score)
        alert_hist.append(alert)
        # 模拟网络抖动：第 90~100 窗 broker 离线（边缘应缓存）
        if 90 <= w < 100:
            broker.online = False
        else:
            if not broker.online and w == 100:
                print(f"[edge] 网络恢复，补传 {len(edge.buffer)} 条缓存（断网期间未丢数据）")
            broker.online = True
            if edge.buffer:                  # 恢复后刷缓存
                while edge.buffer:
                    pb = edge.buffer.pop(0)
                    broker.publish(f"{PLANT}/{DEVICE}/telemetry", pb)
                    edge.sent_features.append(len(pb))

    # 带宽对账
    raw_equiv = edge.sent_raw_equiv
    feat_bytes = sum(edge.sent_features)
    ratio = raw_equiv / max(feat_bytes, 1)
    print(f"\n[带宽] 若直传原始波形 = {raw_equiv/1024:.1f} KB；"
          f"实际只传特征/告警 = {feat_bytes/1024:.2f} KB；"
          f"节省 {ratio:.1f}×")
    print(f"[云] 收到 {len(cloud.rms_hist)} 条消息，生成 {len(cloud.work_orders)} 张工单；"
          f"断网缓存 {len(edge.buffer)} 条未丢")

    # ---- 图 ----
    fig, ax = plt.subplots(2, 1, figsize=(12, 8), sharex=False)

    # 上：特征时间线 + 告警 + 退化区 + 断网区
    x = np.arange(N_WIN)
    ax[0].plot(x, kurt_hist, color="#1f77b4", lw=1.2, label="峭度(kurtosis)")
    ax[0].axhline(edge.kurt_thr[0] + 3 * edge.kurt_thr[1], color="k", ls="--",
                  lw=1.2, label="告警阈值(基线+3σ)")
    ax[0].axvspan(120, N_WIN, color="#d62728", alpha=0.08, label="真实退化区")
    ax[0].axvspan(90, 100, color="#555", alpha=0.12, label="断网缓存区")
    a_idx = np.where(alert_hist)[0]
    if len(a_idx):
        ax[0].scatter(a_idx, np.array(kurt_hist)[a_idx], color="#d62728",
                      zorder=5, s=18, label="边缘告警")
    ax[0].set_ylabel("峭度")
    ax[0].set_title("边缘特征提取 + 告警判定的真实时序")
    ax[0].legend(fontsize=8, loc="upper left")

    # 下：带宽对比（每窗原始 vs 实际特征，累计）
    raw_per = np.full(N_WIN, WIN * 4)         # 假设每窗都直传原始
    cum_raw = np.cumsum(raw_per) / 1024
    # 实际传出字节（心跳~小，完整特征~大）按 seq 近似还原
    cum_feat = np.cumsum([
        (WIN * 4 * 0.02) if (i % THROTTLE_FULL != 0) else (WIN * 4 * 0.12)
        for i in range(N_WIN)
    ]) / 1024
    ax[1].plot(x, cum_raw, color="#d62728", lw=2, label="若直传原始波形(累计 KB)")
    ax[1].plot(x, cum_feat, color="#2ca02c", lw=2, label="只传特征/告警(累计 KB)")
    ax[1].fill_between(x, cum_feat, cum_raw, color="#2ca02c", alpha=0.15)
    ax[1].set_xlabel("窗口序号")
    ax[1].set_ylabel("累计传输量 (KB)")
    ax[1].set_title(f"带宽节省：特征化后约 {ratio:.0f}× 压缩（原始波形 {raw_equiv/1024:.0f}KB → 特征 {feat_bytes/1024:.1f}KB）")
    ax[1].legend(fontsize=8, loc="upper left")
    savefig("edge_cloud_pipeline", "engineering")

    print("\n[结论] 边缘不传原始波形，只传特征+健康分+告警（"
          f"{feat_bytes/1024:.2f}KB vs 原始 {raw_equiv/1024:.1f}KB，省 {ratio:.0f}×）；"
          "MQTT 分层 topic 解耦端云；断网时本地缓存、恢复补传，关键片段零丢失；"
          "云侧聚合后把持续告警收敛成工单。生产环境把 LocalMQTTBroker 换成 "
          "paho-mqtt + Mosquitto/EMQX 即同等架构。")


if __name__ == "__main__":
    main()
