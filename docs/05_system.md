# 模块 05：系统层 / 部署 —— 从训练态到可上线服务

> 对应代码：`src/module_05_system/deploy_demo.py`（自包含训练一个轻量 1D-CNN 并导出）。

## 工程师的分水岭
把模型在 notebook 里跑出 97% 不算本事；能把它**稳定、低延迟、可复现**地部署到边缘设备/工控机里才算。本模块演示端到端链路：

```
训练态 PyTorch 模型
   │  torch.onnx.export  (opset 17)
   ▼
ONNX 中间表示（跨平台、跨框架）
   │  onnxruntime (CPU)
   ▼
边缘设备推理服务（毫秒级，可整合进 PLC / 网关）
```

> 部署架构一图流（边缘推理 + 云端持续学习）：

![PHM 部署架构](figures/illustrations/deploy_arch.png)

## 实测「部署体检卡」（RTX 4090 D 上训练，CPU 推理基准）
| 指标 | 数值 |
|---|---|
| 参数量 | 25,636 |
| PyTorch 权重体积 | 103.3 KB |
| ONNX 模型体积 | 102.2 KB |
| PyTorch-CPU 延迟 | 0.056 ms/样本（≈1.8 万条/秒） |
| ONNX-RT 延迟 | 0.038 ms/样本（≈2.6 万条/秒） |
| ONNX vs PyTorch 输出最大差异 | 3.8e-6（一致性通过） |

## 5.2 INT8 量化：再压一轮、讲清边界
> 对应代码：`src/module_05_system/quantize_demo.py`

训练、导出 ONNX 之后，工业部署还差一步 **量化**。脚本复用上面的 MiniCNN，自包含地对比两种量化策略与真实指标。

![INT8 量化对比卡](figures/system/quant_card.png)

### 实测（静态 INT8，FP32 校准）
| 指标 | FP32 | 静态 INT8 | 说明 |
|---|---|---|---|
| 模型体积 | 102.2 KB | **31.0 KB（3.3×）** | 权重 FP32→INT8，体积≈1/3 |
| 单样本延迟 (CPU) | 0.121 ms | 0.238 ms | 小模型下量化开销抵消计算收益 |
| 吞吐 | 8,257 条/秒 | 4,205 条/秒 | 同上 |
| 测试集准确率 | 0.7266 | 0.7344 | 几乎无损（量化噪声甚至轻微正则化） |
| logits 最大差异 | — | 1.06e-1 | 远小于分类决策边界 |

### 三个必须讲清的工程事实
1. **动态量化的「卷积坑」**：动态量化（Dynamic Quantization）对卷积层生成 `ConvInteger` 节点，而**当前 onnxruntime 的 CPU 执行提供器没有该算子实现**——模型能导出、体积也能压到 33.6 KB，却**创建推理会话即报 `NotImplemented`**。脚本用 `try/except` 如实捕获了这个限制，而不是假装成功。
2. **能跑的 INT8 卷积走静态量化**：用一小批校准数据确定激活范围，生成 `QLinearConv`，CPU EP 原生支持 → 体积 3.3×、精度几乎无损。
3. **量化 ≠ 一定更快**：本例 MiniCNN 仅 2.5 万参数、卷积极轻量，量化前后插入的 Quantize/Dequantize 节点开销盖过了 int8 计算收益，**延迟反而变慢**。真实启示：**小模型量化的首要收益是「省体积/省内存」（对 MCU 是生死线），延迟收益要看是否有真 int8 计算内核、以及模型/Batch 规模**——大 CNN、LSTM、Transformer 上才明显。对 x64 还应优先 `QuantFormat.QDQ` 以启用更优内核。

### 工程启示
1. **ONNX 是部署的事实标准**：一次导出，可在 x86/ARM/嵌入式 NPU 上的 onnxruntime、TensorRT、OpenVINO 跑。
2. **INT8 量化首益在体积**：102KB→31KB（3.3×），MCU/工控机内存只有几十 MB 时这是关键。
3. **动态量化别直接用在有卷积的模型上**：本环境 CPU EP 缺 `ConvInteger`；工业上优先静态量化或选支持 INT8 卷积的 EP（QNN / DML / TensorRT）。
4. **边带推理要配套预处理**：振动窗口截取、归一化必须和训练完全一致，否则现场掉点。
5. **监控比训练更重要**：现场数据分布会漂移，要持续采样本、定期重训、设置信区间告警（呼应模块 04 的 MC Dropout）。

## 一键复现
```bash
python src/module_05_system/deploy_demo.py     # 训练+导出 FP32 ONNX + 部署体检卡
python src/module_05_system/quantize_demo.py    # INT8 静态量化 + 动态量化局限对照
# 导出文件：src/module_05_system/bearing_cnn.onnx  (FP32)
#           src/module_05_system/bearing_cnn.int8.onnx  (静态 INT8，可运行)
```
