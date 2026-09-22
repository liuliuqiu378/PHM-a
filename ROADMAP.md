# PHM 教程项目路线图与进度看板

> 维护项目目标、阶段与进展，方便团队/他人快速理解当前状态和下一步。

## 一、总体目标

在 `shaojie` conda 环境 + RTX 4090 D 本地算力上，搭建一套**可运行、图文结合、从内到外**的 PHM 成长教程，覆盖硬件-数据-算法-系统四层，并完整落地 4 个端到端场景。

## 二、资源盘点

| 资源 | 状态 | 说明 |
|------|------|------|
| 本地 GPU | [完成] | RTX 4090 D 24G，CUDA 可用 |
| conda 环境 | [完成] | `shaojie`，Python 3.10 |
| 包安装 | [完成] | torch 2.5.1+cu121（CUDA 可用）、numpy/scipy/matplotlib/sklearn/pandas/tqdm/h5py |
| CWRU 数据 | [完成] | 161 个 .mat，130 个可读（31 个损坏），含 DE/FE/BA 与 RPM |
| PHM2023/2024/2025 数据 | [完成] | 挑战赛训练/验证/测试 zip + 说明 PDF；PHM2024 已解包，确认是涡桨发动机扭矩裕度任务 |
| 历史实验结果 | [完成] | 已有 Normal vs Inner Race 二分类图（本教程已扩展为 4 类） |

## 三、模块里程碑

| 模块 | 主题 | 状态 | 交付物 | 进展说明 |
|------|------|------|--------|----------|
| 00 | PHM 全景 | [完成] | `draw_overview.py` + 图 + 文档 | 已生成 phm_loop.png |
| 01 | 硬件层：传感器与信号链 | [完成] | `demo_sampling_aliasing.py` + 图 + 文档 | 混叠仿真图已生成 |
| 02 | 数据层：信号处理与特征 | [完成] | `demo_fft_envelope.py` + `signal_utils.py` + 图 + 文档 | 包络解调图已生成 |
| 03 | 算法层：故障诊断 | [完成] | `train_ml.py` + `train_1dcnn.py` + 3 张图 + 文档 | 真实双协议结果见下 |
| 04 | 算法层：RUL 预测 | [完成] | `demo_rul.py` + 3 图 + 文档 | 仿真退化 + LSTM(RMSE29.7) vs 模型法(96.6) + MC Dropout 置信区间 |
| 05 | 系统层：部署与工程化 | [完成] | `deploy_demo.py` + `quantize_demo.py` + ONNX/INT8 + 体检卡 + 文档 | 轴承 CNN→ONNX 102KB/2.6万条/秒；INT8 静态量化 31KB(3.3×)、精度无损、动态量化 ConvInteger 局限已演示 |
| 06 | 端到端场景 | [完成] | 5 个场景完整脚本 + 文档 + 效果展示 | A/B/C/D/E 全部跑通（E=PHM2025 真实 RUL） |
| 07 | 工程实战：数据闭环与异常检测 | [完成] | `anomaly_detection.py` + `edge_cloud_pipeline.py` + `drift_active_learning.py` + `phm_lifecycle` 图 + 文档 | 无监督AE(AUC=1.0理想)；边缘-云管线可跑(22×带宽节省/断网缓存/工单收敛)；采集/传输/端云/训练闭环/异常样本难题文档化；概念漂移+主动学习闭环(主动0.025/随机0.050/不重训0.250) |

## 四、已验证的关键结论（第 3 层，可写进汇报）

CWRU 4 类（Normal/Ball/InnerRace/OuterRace），130 文件，窗口 2048：

| 协议 | RF(物理特征) | 1D-CNN(原始波形) |
|------|--------------|------------------|
| 窗口级(同工况) | 97.7% | 92.6% |
| 文件级(跨工况) | 67.3% | 73.0% |

结论：同工况下特征工程胜出；跨工况下原始 CNN 泛化更好；两者换负载都大幅掉点，点明域适应必要性。

## 五、场景里程碑

| 场景 | 名称 | 状态 | 目标产出 | 当前进展 |
|------|------|------|----------|----------|
| A | 电机轴承故障诊断（CWRU） | [完成] | 完整数据流 + 训练 + 评估 + 部署 | 双协议真实结果(RF97.7%/CNN73%跨工况)；专文档见 docs/scenario_a_bearing.md |
| B | PHM 挑战赛 发动机扭矩裕度 | [完成] | 回归+分类+特征重要性 | 74万样本，MLP R²≈1.0、AUC≈1.0（近确定性，已如实说明） |
| C | 锂电池 SOH/RUL 预测 | [完成] | ECM 仿真+SOH估计+RUL外推 | SOH R²=0.78，RUL 误差44循环（线性可外推，树模型不外推） |
| D | 齿轮箱健康评估 | [完成] | 振动仿真+边带/峭度诊断 | RF 100%，边带占比最重要(0.552) |
| E | PHM2025 发动机 RUL | [完成] | 真实 RUL 回归+留一法 CV | 4 台发动机；WW R²0.38/HPC_SV 0.67/HPT_SV 0.29（真实现场难度） |

## 六、基础设施里程碑

| 任务 | 状态 | 交付物 | 说明 |
|------|------|--------|------|
| 项目目录结构 | [完成] | 见 `README.md` 目录树 | 已创建 |
| 项目 README | [完成] | `README.md` | 已落盘 |
| 路线图/看板 | [完成] | 本文件 | 持续更新 |
| 依赖管理 | [完成] | `requirements.txt` | 已落盘 |
| 教程目录索引 | [完成] | `docs/index.md` + `docs/catalog.md` | 已落盘 |
| MkDocs 发布站点 | [完成] | `mkdocs.yml` + `site/` | material 主题；14 篇文档→15 页；`docs/figures` 符号链接根 `figures`（单一来源）；`mkdocs build` 零警告 |
| 方法总图谱 + 传感器映射 | [完成] | `docs/08_method_landscape.md` + `docs/09_fault_sensor_map.md` | 面试/讲解用收口：方法谱系表+三组易混区别+选型决策树+效果幻觉拆解；失效物理→传感器因果链映射（轴承/齿轮/转子/电机MCSA/油液/电池/发动机）+选型三原则；已入 nav/catalog/首页 |
| 故障机理 + 行业知识 | [完成] | `docs/10_fault_mechanism.md` + `docs/11_industry.md` | 真入门/真壁垒：四大根本机理+退化三阶段+故障频率几何直觉+浴盆曲线；维护四级跳(RCM)+ISO标准谱系+OEE/停机成本+告警疲劳+五大落地壁垒；已入 nav/catalog/首页 |
| 信号处理进阶 | [完成] | `docs/12_signal_advanced.md` + `src/module_02_signal/demo_advanced_signal.py` | 阶次跟踪(变速FFT糊→角域锁阶)、谱峭度/Kurtogram(自动选最优共振带)、AM/FM调制与边带(内圈vs外圈物理依据)；3张图；已入 nav/catalog/首页 |
| 信号处理与物理进阶二 + 应用场景 | [完成] | `docs/13_signal_advanced2.md` + `docs/14_applications.md` + `src/module_02_signal/demo_advanced2_signal.py` | TSA/倒频谱/转子动力学指纹(1×/2×/0.5×/0.42×油膜涡动/临界转速)/Paris-Miner疲劳RUL/循环平稳/HHT(3图)；PHM真实应用场景(应用地图+6行业落地+边缘云工单部署+落地vsPoC)；已入 nav/catalog/首页 |
| 面试与速查 | [完成] | `docs/15_interview_guide.md` + `docs/16_glossary.md` + `docs/17_health_indicator.md` | 面试5分钟提纲(开场+主线+9追问标准回答+保命话)、术语表(六类缩写速查)、HI构建(特征→RUL桥:四种融合+好HI三标准+串02→04)；已入 nav/catalog/首页 |
| 迁移学习/域适应 | [完成] | `docs/18_transfer_learning.md` + `src/module_03_diagnosis/demo_domain_shift.py` | 跨机掉点之谜:现象(RF0.977→0.673)、根因(分布偏移:工况/设备个体FRF/协议骗自己)、demo(绝对RMS跨域50%归一化救回99.9%)、解法谱系(数据增广/归一化+物理频率特征/MMD-DANN/微调+主动学习/AE天然跨机/组合拳)、评测必须跨工况；已入 nav/catalog/首页 |
| 工业案例集 + 理论扩宽 | [完成] | `docs/19_industrial_cases.md` + `docs/20_theory_broaden.md` + `src/module_industry/demo_cases.py` | 资产扩类(往复/结构/电气/电子)、8真实案例(风电齿轮箱/轨交热轴/石化MCSA/航空EGT/变压器DGA/半导体FDC/采矿车队/数据中心)带图(退化趋势+成本对比)、共性表；理论补 DGA三比值/IEC60599、随机退化(Wiener/Gamma/IG)、异常检测谱系、FMEA/FTA/RCA、ISO10816振动A/B/C/D分区、油液/红外/AE/电涡流/超声传感、IEC61508/SIL、数字孪生；已入 nav/catalog/首页 |
| 统一绘图风格 | [完成] | `src/common/plot_style.py` | 含中文字体处理 |
| 公共数据加载器 | [完成] | `src/common/cwru_loader.py` | 4 类映射 + h5py 回退 + 坏文件跳过 |
| 信号处理工具 | [完成] | `src/common/signal_utils.py` | FFT/包络/特征/故障频率 |
| 环境冒烟测试 | [完成] | `tests/test_env.py` | torch/onnx/onnxruntime 均就绪 |
| 概念示意图生成器 | [完成] | `src/module_illustrations/draw_concepts.py` | 7 张概念图→`figures/illustrations/`，配文档建立直觉 |

## 七、更新日志

| 日期 | 更新人 | 内容 |
|------|--------|------|
| 2026-09-20 | CodeBuddy | 创建目录/README/路线图/依赖；确认 CWRU 格式；装环境(cu121)；完成 00~03 层脚本+图+文档；得到轴承诊断双协议真实结果；确定 31 个损坏文件 |
| 2026-09-20 | CodeBuddy | 完成场景 B（PHM2024 发动机扭矩裕度，74万样本 MLP R²≈1.0）、模块5 ONNX 部署（2.6万条/秒）、场景 C 电池 ECM 仿真（SOH R²=0.78）、场景 D 齿轮箱仿真（RF 100%）；补 4 篇场景/系统文档；共 20 张教学图 |
| 2026-09-20 | CodeBuddy | 模块5 INT8 量化（静态 31KB/3.3×、精度无损；动态量化 ConvInteger 局限已演示）；场景 E PHM2025 真实 RUL（4 台发动机、留一法 CV、R²=0.29–0.67）+ 文档；共 24 张教学图 |
| 2026-09-20 | CodeBuddy | 工程实战三 demo（无监督异常检测/边缘-云管线22×/概念漂移+主动学习）；MkDocs 发布站点(mkdocs.yml+material主题，14篇文档→15页，构建零警告)；docs 图片链接改 `figures/`，`docs/figures` 符号链接根 `figures` |
| 2026-09-21 | CodeBuddy | 补 P0 两块收口文档（用户要用于面试/讲解）：方法总图谱(08，谱系表+MLvsDLvs信号处理区别+选型决策树+效果幻觉拆解)、故障机理→传感器映射(09，失效物理因果链+轴承公式+电机MCSA/油液/电池/发动机+选型三原则)；接入 MkDocs nav/catalog/首页，`mkdocs build` 零警告 |
| 2026-09-21 | CodeBuddy | 补 P1 两块（用户强调"故障机理+行业专业知识才是真入门真壁垒、要通俗易懂"）：故障机理深入(10，四大根本机理磨损/疲劳/腐蚀/蠕变+退化三阶段+故障频率几何直觉+各故障生活类比+浴盆曲线+机理-数据-耦合张力)、行业专业知识(11，维护四级跳RCM+ISO标准谱系10816/20816/13374/18436/SAE+ OEE与停机成本+告警疲劳+异常≠可行动+五大落地壁垒+OMEvs第三方+一句话定位)；接 nav/catalog/首页，`mkdocs build` 零警告 |
| 2026-09-21 | CodeBuddy | 落地 P0 信号处理三件套（用户要补"信号+机械物理知识"）：`docs/12_signal_advanced.md` + `src/module_02_signal/demo_advanced_signal.py`(numpy+scipy，3图)。阶次跟踪(变速下FFT糊→按转角重采样锁阶，补教程恒转速不真实假设)、谱峭度/Kurtogram(STFT幅值时间维峭度自动定位最优共振带，把"选对带通靠人试"变自动)、AM/FM调制与边带(内圈随转频AM→BPFI±fr边带、外圈固定→BPFI单线，物理区分内外圈)；接 nav/catalog/首页，`mkdocs build` 零警告 |
| 2026-09-21 | CodeBuddy | 补 P1/P2 信号物理专题 + PHM应用场景（用户要"简单易懂讲解"且"实际应用场景讲得少"）：`docs/13_signal_advanced2.md`(同步平均TSA/倒频谱/转子动力学指纹[1×不平衡,2×不对中,0.5×松碰,0.42×油膜涡动,临界转速]/Paris-Miner疲劳RUL物理/循环平稳/HHT-EMD，3图) + `docs/14_applications.md`(应用地图[旋转/热力/电化学]、风电/轨交/航空/石化/制造/电池六行业落地细节+传感器方法ROI、边缘-云-工单部署形态、真实落地vsPoC讲真话)。`src/module_02_signal/demo_advanced2_signal.py`(numpy+scipy,3图:tsa/cepstrum/rotor)。接 nav/catalog/首页，`mkdocs build` 零真实警告 |
| 2026-09-21 | CodeBuddy | 整体回顾后补面试/速查三篇（用户要"整体回顾还有啥补"）：`docs/15_interview_guide.md`(5分钟讲法:开场白+主线[一个轴承串00~14]+方法谱系收口+9条高频追问标准回答[准确率/FFT不够/带通/内外圈/无标签/漂移/RUL/应用/落地难/MCSA]+收尾+临场保命话)、`docs/16_glossary.md`(六类缩写速查含文档锚点)、`docs/17_health_indicator.md`(HI=特征→RUL桥:四种融合[物理/PCA-AE/距离/专家]+好HI三标准[单调/可预测/可解释]+串02→04+诚实边界)。接 nav(新增"面试与速查"节)/catalog/首页，`mkdocs build` 零真实警告，15/16/17 HTTP 200 |
| 2026-09-22 | CodeBuddy | 补迁移学习/域适应专章（用户指"RF0.977→0.673跨工况暴跌是关键"要讲原因+解决）：`docs/18_transfer_learning.md` + `src/module_03_diagnosis/demo_domain_shift.py`(numpy+sklearn合成域偏移,1图+两准确率)。现象(引03的RF97.7%→67.3%/CNN92.6%→73.0%跨工况暴跌,窗口级泄漏高估)、根因(分布偏移covariate shift:工况负载变→绝对幅值变/设备个体FRF传递路径不同/协议骗自己)、demo(绝对RMS特征源域100%→跨域50%暴跌;归一化峰值/RMS跨域99.9%救回,图示两域分布错开vs重叠)、解法谱系(数据增广域随机化/幅值归一化+物理频率特征BPFI负载无关PHM独门优势/MMD-CORAL-DANN对抗域适应/自训练伪标签/微调+主动学习/AE无监督天然跨机/物理特征冷启动+DL精细化组合拳)、评测必须跨文件跨工况跨机+留一法、面试话术。接 nav/catalog/首页，`mkdocs build` 零真实警告，18 HTTP 200 |
| 2026-09-22 | CodeBuddy | 用户要"补理论+行业知识、扩宽场景+实际工业案例"：补 `docs/19_industrial_cases.md`(资产扩类:往复/结构/电气/电子 四类，超出原旋转/热力/电池；8个真实工业案例=风电齿轮箱CMS预警/轨交热轴声学/石化泵MCSA/航空EGT气路/变压器DGA三比值/半导体FDC/采矿车队HI+域适应/数据中心冷水机+硬盘SMART，每个含资产→传感器→方法→量级结果→教训，含图case_wind_gearbox_trend.png退化趋势+成本对比，共性表+面试用法) + `docs/20_theory_broaden.md`(DGA三比值IEC60599、随机退化Wiener/Gamma/IG补04、异常检测谱系补07、FMEA/FTA/RCA知识型诊断、ISO10816振动A/B/C/D分区mm/s、油液/红外/AE/电涡流/超声传感、IEC61508/SIL功能安全、数字孪生三层价值) + `src/module_industry/demo_cases.py`(numpy合成退化+成本图)。接 nav/catalog/首页，`mkdocs build` 零真实警告，19/20 HTTP 200 |

## 八、下一步计划

1. `mkdocs gh-deploy` 推 GitHub Pages，或 `mkdocs build` 后把 `site/` 托管到任意静态空间（已具备）；
2. 用 LSTM/Transformer 把场景 E 升级为「发动机内时间序列」退化建模；
3. 视情况加 PHM2023 数据挑战赛场景或电池真实数据集(NASA)对照；
4. INT8 量化可补 QDQ 格式对比（x64 延迟更优）；
5. 可选导出 PDF（mkdocs-with-pdf）做离线版。
