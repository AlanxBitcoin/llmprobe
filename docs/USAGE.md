# Usage Guide

## 1. 项目用途

这个项目用于对本地 `Meta-Llama-3-8B-Instruct` 做两类分析：

- 单个英文单词的第 8 层隐藏状态分析
- 多个英文单词的全层传播分析

同时项目支持：

- 100 词批处理
- 属性 probe 训练
- 单词属性预测
- 图片、JSON、CSV、HTML、MP4 导出

## 2. 环境要求

- Windows + PowerShell
- Python 已安装
- 本地 Hugging Face 兼容模型目录
- 推荐模型：`Meta-Llama-3-8B-Instruct`

当前默认配置使用的模型路径在：

[default.yaml](D:\vbcode2026\LLM探针\configs\default.yaml)

如果模型路径变化，请修改：

```yaml
model:
  model_name_or_path: "你的本地模型目录"
```

## 3. 依赖安装

在项目根目录执行：

```powershell
pip install -r requirements.txt
```

如果你已经在当前环境中完成安装，可跳过。

## 4. 配置文件

项目当前有两份主要配置：

- 默认稳定配置：
  [default.yaml](D:\vbcode2026\LLM探针\configs\default.yaml)
- 4bit 配置：
  [quantized-4bit.yaml](D:\vbcode2026\LLM探针\configs\quantized-4bit.yaml)

推荐先用默认配置，确认流程正常后再尝试 4bit。

## 5. 常用命令

### 5.1 单个单词分析

```powershell
python main.py run-single-word apple
```

输出：

- 图片：
  [outputs/single_word/figures](D:\vbcode2026\LLM探针\outputs\single_word\figures)
- JSON：
  [outputs/single_word/data](D:\vbcode2026\LLM探针\outputs\single_word\data)

示例文件：

- [apple.png](D:\vbcode2026\LLM探针\outputs\single_word\figures\apple.png)
- [apple.json](D:\vbcode2026\LLM探针\outputs\single_word\data\apple.json)

### 5.2 多个单词分析

```powershell
python main.py run-multi-word apple banana orange
```

输出：

- 图片：
  [outputs/multi_word/figures](D:\vbcode2026\LLM探针\outputs\multi_word\figures)
- JSON：
  [outputs/multi_word/data](D:\vbcode2026\LLM探针\outputs\multi_word\data)

### 5.3 单词批处理

```powershell
python main.py run-single-batch
```

该命令会读取：

[word_list_100.txt](D:\vbcode2026\LLM探针\data\word_list_100.txt)

并逐词执行单词分析。

输出目录：

[outputs/single_batch](D:\vbcode2026\LLM探针\outputs\single_batch)

关键结果：

- 索引页：
  [index.html](D:\vbcode2026\LLM探针\outputs\single_batch\index.html)
- 总览页：
  [overview.html](D:\vbcode2026\LLM探针\outputs\single_batch\overview.html)
- 汇总表：
  [summary.csv](D:\vbcode2026\LLM探针\outputs\single_batch\summary.csv)
- 录屏：
  [single_batch.mp4](D:\vbcode2026\LLM探针\outputs\single_batch\single_batch.mp4)

### 5.4 多词批处理

```powershell
python main.py run-multi-batch --batch-size 3
```

表示每次读取 3 个词做一组传播分析。

输出目录：

[outputs/multi_batch_3](D:\vbcode2026\LLM探针\outputs\multi_batch_3)

关键结果：

- 索引页：
  [index.html](D:\vbcode2026\LLM探针\outputs\multi_batch_3\index.html)
- 总览页：
  [overview.html](D:\vbcode2026\LLM探针\outputs\multi_batch_3\overview.html)
- 汇总表：
  [summary.csv](D:\vbcode2026\LLM探针\outputs\multi_batch_3\summary.csv)
- 录屏：
  [multi_batch_3.mp4](D:\vbcode2026\LLM探针\outputs\multi_batch_3\multi_batch_3.mp4)

### 5.5 对每张词表做全局分析

```powershell
python main.py run-global-analysis-all
```

默认会自动发现：

- [data](D:\vbcode2026\LLM探针\data) 目录下所有 `.txt` 词表

并对每张词表执行：

- 单词批处理全局分析
- 多词批处理全局分析
- 维度报告

聚合输出目录：

- [outputs/global_analysis_all](D:\vbcode2026\LLM探针\outputs\global_analysis_all)

关键结果：

- 聚合索引页：
  [index.html](D:\vbcode2026\LLM探针\outputs\global_analysis_all\index.html)
- 聚合总览页：
  [overview.html](D:\vbcode2026\LLM探针\outputs\global_analysis_all\overview.html)
- 汇总表：
  [summary.csv](D:\vbcode2026\LLM探针\outputs\global_analysis_all\summary.csv)

如果只想分析指定词表，可以显式传入文件：

```powershell
python main.py run-global-analysis-all data/word_list_100.txt data/english_punctuation.txt
```

如果想修改发现范围，可以使用：

```powershell
python main.py run-global-analysis-all --word-dir data --glob *.txt --batch-size 2
```

### 5.6 主类别线性 Probe

```powershell
python main.py run-probe
```

输出目录：

[outputs/probe](D:\vbcode2026\LLM探针\outputs\probe)

关键结果：

- 指标：
  [probe_metrics.json](D:\vbcode2026\LLM探针\outputs\probe\probe_metrics.json)
- 预测明细：
  [probe_predictions.csv](D:\vbcode2026\LLM探针\outputs\probe\probe_predictions.csv)

### 5.7 属性 Probe

```powershell
python main.py run-attribute-probe
```

输出目录：

[outputs/attribute_probe](D:\vbcode2026\LLM探针\outputs\attribute_probe)

关键结果：

- 指标：
  [attribute_probe_metrics.json](D:\vbcode2026\LLM探针\outputs\attribute_probe\attribute_probe_metrics.json)
- 汇总：
  [attribute_probe_summary.csv](D:\vbcode2026\LLM探针\outputs\attribute_probe\attribute_probe_summary.csv)
- 预测明细：
  [attribute_probe_predictions.csv](D:\vbcode2026\LLM探针\outputs\attribute_probe\attribute_probe_predictions.csv)

### 5.8 单词属性预测

```powershell
python main.py predict-attributes apple
```

输出文件：

- [predict_apple.json](D:\vbcode2026\LLM探针\outputs\predict_apple.json)

## 6. 如何看结果

### 6.1 单词图怎么看

单词图通常包含四部分：

- 第 8 层 top-k 激活维度
- 概念类型相似度
- 概念值相似度
- 属性预测卡片/条形图

建议重点关注：

- `pred_category`
- `pred_color`
- `pred_shape`
- `pred_taste`

### 6.2 单词批处理索引页怎么看

单词批处理的：

[index.html](D:\vbcode2026\LLM探针\outputs\single_batch\index.html)

支持：

- 看缩略图
- 看核心属性标签
- 按 `word/category/color/shape/taste` 过滤
- 按最小置信度过滤
- 只看任意核心属性高于阈值的结果

### 6.3 单词总览页怎么看

单词总览页：

[overview.html](D:\vbcode2026\LLM探针\outputs\single_batch\overview.html)

适合快速看：

- 高频类别
- 高频颜色
- 高频形状
- 高频味道
- 高置信度预测
- 低置信度复核对象

### 6.4 多词传播图怎么看

多词传播图一般包含：

- 层-词热力图
- 简化传播线路图

建议重点关注：

- strongest layer
- 哪些词在高层持续保持高响应
- 哪些词之间出现明显跨层连接

### 6.5 多词总览页怎么看

多词总览页：

[overview.html](D:\vbcode2026\LLM探针\outputs\multi_batch_3\overview.html)

适合快速看：

- strongest layer 分布
- strongest batch
- review candidates

## 7. 数据文件说明

主要数据文件位于：

[data](D:\vbcode2026\LLM探针\data)

包括：

- 测试词表：
  [word_list_100.txt](D:\vbcode2026\LLM探针\data\word_list_100.txt)
- 概念类型词表：
  [concept_types.txt](D:\vbcode2026\LLM探针\data\concept_types.txt)
- 概念值词表：
  [concept_values.txt](D:\vbcode2026\LLM探针\data\concept_values.txt)
- 概念目录：
  [concept_catalog.yaml](D:\vbcode2026\LLM探针\data\concept_catalog.yaml)
- 主类别标签：
  [word_labels.csv](D:\vbcode2026\LLM探针\data\word_labels.csv)
- 属性标签：
  [word_attributes.csv](D:\vbcode2026\LLM探针\data\word_attributes.csv)

## 8. 推荐使用顺序

推荐按这个顺序使用：

1. 先跑一个单词：
```powershell
python main.py run-single-word apple
```

2. 再跑属性预测：
```powershell
python main.py predict-attributes apple
```

3. 再跑单词批处理：
```powershell
python main.py run-single-batch
```

4. 查看：
- [index.html](D:\vbcode2026\LLM探针\outputs\single_batch\index.html)
- [overview.html](D:\vbcode2026\LLM探针\outputs\single_batch\overview.html)

5. 最后跑多词传播：
```powershell
python main.py run-multi-batch --batch-size 3
```

## 9. 已知说明

- 默认配置更稳，4bit 配置更省资源但可能更慢或更敏感
- 当前属性预测中，`category / shape / taste` 通常比 `color / emotion / material` 更稳定
- 多词传播图目前是“简化线路图”，不是严格神经回路级因果分析

## 10. 后续可扩展方向

- 增强多词传播图
- 增强属性 probe
- 增强总览统计
- 增加更多测试词和标签
- 增加更细粒度的 neuron/head 分析
