# PHM 术语表（速查）

> 教程里出现的缩写与黑话一页速查。面试/讲解前扫一遍，被问到缩写能立刻接上。
> 方括号里是首次详细讲解的文档。

## 总览与策略
| 缩写 | 全称 | 含义 |
|---|---|---|
| PHM | Prognostics and Health Management | 故障预测与健康管理（本教程主题） |
| CBM / PdM | Condition-Based / Predictive Maintenance | 状态维修 / 预测性维修（按需修） |
| RCM | Reliability-Centered Maintenance | 以可靠性为中心的维修（先分级再决定上不上 PHM）[`11`](./11_industry.md) |
| OEE | Overall Equipment Effectiveness | 设备综合效率 = 可用率×性能×质量[`11`](./11_industry.md) |

## 故障与物理
| 缩写 | 含义 | 见 |
|---|---|---|
| BPFI / BPFO | 轴承内圈/外圈故障频率 | [`09`](./09_fault_sensor_map.md) |
| BSF / FTF | 滚动体/保持架故障频率 | [`09`](./09_fault_sensor_map.md) |
| MCSA | Motor Current Signature Analysis 电机电流特征分析（非侵入式抓转子断条） | [`09`](./09_fault_sensor_map.md) |
| ODM | Oil Debris / Wear Metal 油液磨屑监测 | [`09`](./09_fault_sensor_map.md) |
| TSA | Time Synchronous Averaging 同步平均（用转速当节拍器消噪声） | [`13`](./13_signal_advanced2.md) |
| EMD / HHT | 经验模态分解 / 希尔伯特-黄变换（数据自拆层+瞬时频率） | [`13`](./13_signal_advanced2.md) |
| Paris / Miner | 裂纹扩展定律 / 累积损伤准则（RUL 物理根） | [`13`](./13_signal_advanced2.md) |
| HI | Health Indicator 健康指标（多特征融合成的一条退化曲线） | [`17`](./17_health_indicator.md) |

## 信号处理
| 缩写 | 含义 | 见 |
|---|---|---|
| FFT | 快速傅里叶变换（频域） | [`02`](./02_signal.md) |
| STFT | 短时傅里叶变换（时频） | [`02`](./02_signal.md) |
| SK / Kurtogram | 谱峭度 / 快速谱峭度图（自动选包络带通） | [`12`](./12_signal_advanced.md) |
| AM / FM | 调幅 / 调频（边带来源） | [`12`](./12_signal_advanced.md) |
| Cepstrum | 倒频谱（边带间距→倒频峰） | [`13`](13_signal_advanced2.md) |
| Order Tracking | 阶次跟踪（变速下按转角重采样） | [`12`](./12_signal_advanced.md) |
| Cyclostationarity | 循环平稳（比包络更鲁棒的调制提取） | [`13`](./13_signal_advanced2.md) |

## 数据与模型
| 缩写 | 含义 | 见 |
|---|---|---|
| RUL | Remaining Useful Life 剩余使用寿命 | [`04`](./04_prognostics.md) |
| SOH | State of Health 电池健康度 | [`04`](./04_prognostics.md) |
| AE | AutoEncoder 自编码器（无监督异常检测） | [`03`](./03_diagnosis.md)[`07`](./07_engineering.md) |
| 1D-CNN | 一维卷积网络（直接吃波形） | [`03`](./03_diagnosis.md) |
| ONNX / INT8 | 模型交换格式 / 8位量化（边缘部署） | [`05`](./05_system.md) |
| MC Dropout | 蒙特卡洛 Dropout（RUL 不确定性） | [`04`](./04_prognostics.md) |
| Active Learning | 主动学习（挑最不确定样本请专家标） | [`07`](./07_engineering.md) |
| Concept Drift | 概念漂移（上线后数据分布变） | [`07`](./07_engineering.md) |

## 标准与系统
| 缩写 | 含义 | 见 |
|---|---|---|
| ISO 10816 / 20816 | 机械振动烈度评级（振算不算大） | [`11`](./11_industry.md) |
| ISO 13374 | PHM 数据处理链（≈本教程分层） | [`11`](./11_industry.md) |
| ISO 18436 | 振动分析师认证（人员能力） | [`11`](./11_industry.md) |
| SAE JA6268 | 航空 PHM 指南 | [`11`](./11_industry.md) |
| MQTT / OPC-UA / Kafka | 边缘-云传输协议 | [`07`](./07_engineering.md) |
| CMMS | 计算机化维修管理系统（工单闭环） | [`07`](./07_engineering.md) |
| Digital Twin | 数字孪生（物理设备的虚拟镜像） | （见 [`14`](./14_applications.md) 部署形态） |

## 数据集
| 名称 | 是什么 | 见 |
|---|---|---|
| CWRU | Case Western 轴承台架（12k/48k 振动，教学标准） | [`01`](./01_hardware.md)[`03`](./03_diagnosis.md) |
| PHM2025 | 4 台真实发动机 RUL（近现场、难） | [`04`](./04_prognostics.md) |
| C-MAPSS | NASA 航空发动机退化仿真（RUL 经典） | [`04`](./04_prognostics.md) |
