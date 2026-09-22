# PHM 教程总目录

> 按「由内到外、由理论到实际」的顺序排列，每个模块包含一篇 Markdown 文档和一组可运行的 Python 脚本。所有脚本生成的图统一保存到 `figures/`。

## 学习路径

```text
第 0 层  PHM 全景
    │
第 1 层  硬件层：传感器、信号链、ADC
    │
第 2 层  数据层：时域/频域、包络、特征
    │
第 3 层  算法-诊断：ML + DL
    │
第 4 层  算法-预测：RUL
    │
第 5 层  系统层：部署、数据流、演示
    │
第 6 层  端到端场景
```

## 模块文档

| 编号 | 文档 | 脚本目录 | 核心内容 |
|------|------|----------|----------|
| 00 | [PHM 全景](./00_big_picture.md) | - | PHM 是什么、技术闭环、产业价值、学习地图 |
| 01 | [硬件层：传感器与信号链](./01_hardware.md) | `src/module_01_hardware/` | 传感器类型、IEPE/ICP、ADC、采样定理、混叠仿真 |
| 02 | [数据层：信号处理与特征](./02_signal.md) | `src/module_02_signal/` | FFT、包络解调、STFT/小波、特征工程、CWRU 可视化 |
| 12 | [信号处理进阶：阶次/谱峭度/调制](./12_signal_advanced.md) | `src/module_02_signal/demo_advanced_signal.py` | 阶次跟踪(变速)、谱峭度/Kurtogram(自动选带通)、AM/FM调制与边带(内圈vs外圈物理依据) |
| 13 | [信号处理与物理进阶二：TSA/倒频谱/转子/疲劳/HHT](./13_signal_advanced2.md) | `src/module_02_signal/demo_advanced2_signal.py` | 同步平均TSA、倒频谱、转子动力学指纹(1×/2×/0.5×/0.42×油膜涡动/临界转速)、Paris/Miner(RUL物理)、循环平稳、HHT/EMD |
| 03 | [算法层：故障诊断](./03_diagnosis.md) | `src/module_03_diagnosis/` | SVM/RF/KNN、1D-CNN、评估指标、混淆矩阵 |
| 04 | [算法层：RUL 预测](./04_prognostics.md) | `src/module_04_prognostics/` | 退化建模、LSTM/Transformer、相似度、置信区间 |
| 05 | [系统层：部署与工程化](./05_system.md) | `src/module_05_system/` | 边缘/云、模型服务化、INT8 量化、PHM 演示系统 |
| 07 | [工程实战：数据闭环与异常检测](./07_engineering.md) | `src/module_06_engineering/` | 采集/传输/端云部署/训练闭环/无监督异常检测/边缘-云管线/概念漂移+主动学习 |
| 08 | [方法总图谱（选型指南）](./08_method_landscape.md) | - | 方法谱系表(物理/信号/ML/DL/无监督/RUL/漂移)、三组易混区别、选型决策树、效果幻觉拆解 |
| 09 | [故障机理 → 传感器映射](./09_fault_sensor_map.md) | - | 失效物理→敏感物理量→传感器→特征 因果链；轴承/齿轮/转子/电机MCSA/油液/电池/发动机映射 + 选型三原则 |
| 10 | [故障机理深入（真入门）](./10_fault_mechanism.md) | - | 四大根本机理(磨损/疲劳/腐蚀/蠕变)、退化三阶段、故障频率几何直觉、各故障类比、浴盆曲线、机理-数据-耦合张力 |
| 11 | [行业专业知识（真壁垒）](./11_industry.md) | - | 维护四级跳(RCM)、ISO标准谱系(10816/13374/18436/SAE)、OEE与停机成本、告警疲劳、异常≠可行动、五大落地壁垒、OEM vs 第三方 |\n| 14 | [PHM 真实应用场景（行业落地）](./14_applications.md) | - | 应用地图(旋转/热力/电化学)、风电/轨交/航空/石化/制造/电池六行业落地细节、部署形态(边缘-云-工单)、真实落地 vs PoC 讲真话 |\n| 15 | [面试/讲解 5 分钟讲法提纲](./15_interview_guide.md) | - | 开场白+主线(一个轴承串全场)+方法谱系收口+9条高频追问标准回答+收尾+临场保命话 |
| 16 | [PHM 术语表（速查）](./16_glossary.md) | - | 总览/故障/信号/数据/标准/数据集 六类缩写速查(含文档锚点) |
| 17 | [健康指标 HI 构建（特征→RUL 的桥）](./17_health_indicator.md) | - | HI定义/四种融合路子(物理PCA-AE距离专家)/好HI三标准(单调可预测可解释)/串起02→04 |
| 18 | [迁移学习/域适应（跨机掉点之谜）](./18_transfer_learning.md) | `src/module_03_diagnosis/demo_domain_shift.py` | 现象(RF0.977→0.673跨工况暴跌)、根因(分布偏移:工况/设备个体FRF/协议骗自己)、demo(绝对RMS掉到50%归一化救回99.9%)、解法谱系(数据增广/归一化+物理频率特征/MMD-DANN/微调+主动学习/AE天然跨机/组合拳)、评测必须跨工况 |
| 19 | [工业案例集（真实落地+资产扩类）](./19_industrial_cases.md) | `src/module_industry/demo_cases.py` | 资产扩类(往复/结构/电气/电子)、8真实案例(风电齿轮箱/轨交热轴/石化MCSA/航空EGT/变压器DGA/半导体FDC/采矿车队/数据中心)、图(退化趋势+成本对比)、共性表+面试用法 |
| 20 | [理论扩宽（DGA/随机RUL/FMEA/功能安全/孪生）](./20_theory_broaden.md) | - | DGA三比值法、随机退化(Wiener/Gamma/IG)、异常检测谱系、FMEA/故障树/RCA、ISO10816振动分区(A/B/C/D)、油液/红外/声发射/电涡流/超声传感、IEC61508/SIL、数字孪生 |
| 06 | [场景 A 轴承](./scenario_a_bearing.md) · [B 发动机](./scenario_b_engine.md) · [C 电池](./scenario_c_battery.md) · [D 齿轮箱](./scenario_d_gearbox.md) · [E RUL](./scenario_e_phm2025.md) | `src/scenarios/` | 5 个完整项目（每场景独立文档） |

## 4 个端到端场景

| 场景 | 入口脚本 | 文档 | 说明 |
|------|----------|------|------|
| A：电机轴承故障诊断 | `src/scenarios/scenario_bearing/run_all.py` | [scenario_a_bearing.md](./scenario_a_bearing.md) | CWRU 数据 → 4 类分类 → 1D-CNN → ONNX 部署 |
| B：PHM 挑战赛 发动机扭矩裕度 | `src/scenarios/scenario_engine/train.py` | [scenario_b_engine.md](./scenario_b_engine.md) | 74万样本 → 回归+分类 → 特征重要性 |
| C：锂电池 SOH/RUL | `src/scenarios/scenario_battery/sim.py` | [scenario_c_battery.md](./scenario_c_battery.md) | ECM 仿真 → SOH 估计 → RUL 外推 |
| D：齿轮箱健康评估 | `src/scenarios/scenario_gearbox/sim.py` | [scenario_d_gearbox.md](./scenario_d_gearbox.md) | 振动仿真 → 边带/峭度 → 故障诊断 |
| E：PHM2025 发动机 RUL | `src/scenarios/scenario_phm2025/train.py` | [scenario_e_phm2025.md](./scenario_e_phm2025.md) | 真实 RUL 回归 → 留一法 CV → 特征重要性 |

## 配套代码约定

- 所有脚本默认在 `PHM/` 根目录下执行：`python src/xxx/xxx.py`；
- 脚本生成的图保存到 `figures/xxx.png`；
- 统一使用 `src/common/plot_style.py` 的绘图风格，保证图的风格一致；
- 可重复：随机种子固定，结果可复现。
- 概念示意图（轴承几何/信号链/包络解调/退化-RUL/CNN 结构/齿轮啮合/部署架构）统一由 `src/module_illustrations/draw_concepts.py` 生成，保存到 `figures/illustrations/`；文档里用 `figures/illustrations/xxx.png` 引用（已从 `docs/` 子目录正确解析）。
