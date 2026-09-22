# 场景 A：电机轴承故障诊断（端到端落地）

> 对应代码：`src/scenarios/scenario_bearing/run_all.py`、模块 02/03。
> 数据：本地 `data/CWRU`（130 个可用 `.mat`，31 个损坏已跳过）。

## 一句话讲清这条线
传感器（加速度计）→ 原始振动 → 特征工程 / 原始波形 → 分类模型 → 「正常 / 内圈 / 外圈 / 滚动体」故障判定，最后把模型导出成 ONNX 上边缘设备（见模块 05）。

> 先建立画面：

![轴承剖面与缺陷位置](figures/illustrations/bearing_geometry.png)

![1D-CNN 结构示意](figures/illustrations/cnn_arch.png)

![PHM 部署架构](figures/illustrations/deploy_arch.png)

## 关键结论（实测）
| 协议 | RF(物理特征) | 1D-CNN(原始波形) |
|---|---|---|
| 窗口级（同工况） | **97.7%** | 92.6% |
| 文件级（跨工况） | 67.3% | **73.0%** |

**这是整本教程最重要的一个「反直觉」结论**：
- 同工况下，「人工特征 + 随机森林」吊打小 CNN（97.7% vs 92.6%）——因为故障信息在频域很清晰，物理特征直接抓住了。
- 但**换负载（跨工况）后，原始 CNN 反而比手工特征泛化更好（73% > 67%）**，而且两者都大幅掉点。这恰恰说明：真实现场要的是**域适应 / 迁移学习**，而不是把 CWRU 上 99% 当真理。

## 教学要点
1. 故障特征频率：6205 轴承 BPFI≈162Hz、BPFO≈107Hz，包络谱里能直接看到尖峰（`demo_fft_envelope.py`）。
2. 标签体系：Normal / Ball / InnerRace / OuterRace 四类，覆盖多负载。
3. 不要迷信 99%：CWRU 是同分布「考试题」，现场是「开卷不同题」。

## 一键复现
```bash
python src/scenarios/scenario_bearing/run_all.py
```
