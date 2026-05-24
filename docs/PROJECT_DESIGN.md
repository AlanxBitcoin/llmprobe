# LLM Probe 项目全局设计文档

**版本**: 1.10
**最后更新**: 2026-05-24
**作用**: 项目的单一真实来源 (Single Source of Truth)
**使用说明**: 修改项目时，先修改此文档，然后根据此文档的要求修改代码

**本次更新 (v1.10)**:
- ✅ 确认 `outputs/` 文件夹位置：从项目根目录迁移到 `data/outputs/`
- ✅ 统一输出路径管理：所有输出结果统一存放在 `data/outputs/` 下
- 🔄 代码路径收尾：补齐剩余硬编码路径（`scripts/analysis/analyze_dims.py`、`scripts/analysis/analyze_skip_results.py`、`scripts/analysis/show_head_top10.py`、`scripts/analysis/layer_skip_probe.py`）
- ✅ 完成判定：Python 代码中不再使用旧文件系统路径 `outputs/...`（`/outputs/...` URL 路由与历史文档示例除外）
- 🔄 目录规范收尾：入口与模型操作文件统一在 `src/` 根目录，项目根目录不保留 `.py` 文件

---

## 📋 目录

1. [项目概述](#项目概述)
2. [核心目标](#核心目标)
3. [项目架构](#项目架构)
4. [核心模块](#核心模块)
5. [数据流](#数据流)
6. [配置系统](#配置系统)
7. [API 接口](#api-接口)
8. [模块依赖图](#模块依赖图)
9. [扩展机制](#扩展机制)

---

## 项目概述

### 简介
这是一个 **LLM 隐藏状态探针 (LLM Probe)** 项目，专门用于分析 Llama-3-8B 模型的内部表示和隐藏状态。通过多种分析模式，该项目可以：
- 提取模型各层的隐藏状态
- 分析隐藏状态与概念的对应关系
- 训练线性探针来预测符号属性
- 可视化激活流程和神经元行为

### 技术栈
- **框架**: PyTorch + Transformers
- **模型**: Llama-3-8B-Instruct (可选量化)
- **数据处理**: PyYAML, Pandas, NumPy
- **本地网页 UI**: FastAPI/ASGI + HTML/CSS/JavaScript（本地 server）
- **输出**: CSV 表格数据（Probe 自定义格式）+ MP4 视频（可选）

### 关键特性
- ✅ 支持 4-bit/8-bit 量化
- ✅ 灵活的隐状态提取策略（单词/批量/序列模式）
- ✅ 线性探针训练与评估
- ✅ 概念匹配与符号属性分析
- ✅ 批处理与全局分析
- ✅ CSV 输出模式：每个 Probe 定义自己的 CSV 表格格式
- ✅ 模型运行时 API：模型加载到 GPU 后常驻，多次 study/API 调用优先复用
- ✅ hooks 参数回退机制：临时参数修改必须在调用结束后恢复
- ✅ 本地网页 UI：通过按钮和参数表单调用 Study/Probe，并展示 CSV 与图表
- ✅ 双服务启动：启动时同时启动 Llama API 和网页 server
- ✅ 无参数启动：默认读取 `configs/custom.yaml`
- ✅ 视频可视化生成

---

## 核心目标

### 一级目标（研究目标）
1. **理解模型的表示学习过程**: 通过分析各层隐藏状态，理解模型如何逐步构建高级语义
2. **符号属性探针**: 训练探针预测词的符号属性（如 color, shape 等）
3. **神经元级分析**: 分析哪些神经元对特定概念敏感
4. **层间信息流**: 理解信息如何在网络各层间流动和转换

### 二级目标（技术目标）
1. **隐状态提取**: 提供三种模式的隐状态提取（见[数据流](#数据流)）
2. **参数修改与钩子**: 支持直接操作模型（关闭头、跳过层、注入隐状态、临时修改参数等），并保证临时修改可回退
3. **可重复性**: 所有配置可保存，结果可复现
4. **可扩展性**: 支持添加新的分析脚本、探针和 UI 操作入口

---

## 项目架构

### 目录结构（按文件类型分类）

```
probe/
├── data/                            # 📊 所有数据和输出
│   ├── concept_catalog.yaml         # 概念目录
│   ├── word_*.txt              # 词表
│   ├── word_attributes.csv          # 词属性标注
│   └── outputs/                     # 📁 输出结果（v1.8更新：已从robe根目录迁移到此处）
│       ├── shape_words/             # 形状相关实验输出
│       ├── color_words/          # 颜色相关实验输出
│       └── ...                      # 其他实验输出
│
├── docs/                            # 📖 文档和依赖
│   ├── PROJECT_DESIGN.md            # 🎯 项目全局设计文档（THIS FILE）
│   ├── ARCHITECTURE.md              # 架构详细说明
│   ├── README.md                    # 快速开始
│   ├── HOOKS_USAGE.md               # hooks.py 使用指南
│   ├── requirements.txt             # Python 依赖
│   └── ...其他文档
│
├── configs/                         # ⚙️ 配置文件
│   ├── custom.yaml                  # 默认启动配置（由原 default.yaml 内容复制而来）
│   ├── quantized-4bit.yaml          # 可选量化配置
│   └── ...其他实验配置
│
├── src/                             # 📦 核心代码库
│   ├── __init__.py
│   ├── main.py                      # 🎯 主入口（同时启动 Llama API 和 Web UI server）
│   │
│   ├── 【第1-2层：模型运行时 API】
│   ├── model_loader.py              # 模型加载、缓存、复用与必要重载
│   ├── runtime_api.py               # Llama API 服务层：统一暴露常驻模型调用接口
│   ├── config.py                    # 配置管理
│   ├── hooks.py                     # 🔌 模型操作钩子
│   ├── extract_hidden.py            # 隐状态提取
│   ├── utils.py                     # 工具函数
│   │
│   ├── 【第3层：Probe 定义】
│   ├── probes/                      # Probe 类型（每种定义自己的 CSV 格式）
│   │   ├── __init__.py
│   │   ├── base_probe.py            # Probe 基类
│   │   ├── linear_probe.py          # 线性探针 → CSV_Linear_Probe 格式
│   │   ├── attribute_probe.py       # 属性探针 → CSV_Attribute_Probe 格式
│   │   ├── concept_match.py         # 概念匹配 → CSV_Concept_Match 格式
│   │   └── symbolic_attributes.py   # 符号属性计算
│   │
│   ├── 【第4层：Study 研究脚本】
│   ├── study/                       # 研究/实验脚本（可调节 Probe 参数）
│   │   ├── __init__.py
│   │   ├── study_linear_probe.py    # 线性探针研究脚本
│   │   ├── study_attribute_probe.py # 属性探针研究脚本
│   │   └── ...其他 study 脚本
│   │
│   ├── 【第5层：UI 本地网页服务】
│   ├── ui/                          # 本地 Web UI（调用第4层 Study 和第3层 Probe）
│   │   ├── __init__.py
│   │   ├── server.py                # 启动本地 server，提供 http://127.0.0.1:PORT 页面
│   │   ├── routes.py                # 页面路由和执行 API
│   │   ├── registry.py              # Study/Probe 按钮注册表
│   │   ├── forms.py                 # 参数表单 schema
│   │   ├── result_render.py         # CSV/图表结果渲染
│   │   ├── templates/               # HTML 页面模板
│   │   └── static/                  # CSS/JS 静态资源
│   │
│   ├── 【软件测试：真正 test 含义】
│   ├── test/                        # 真正的软件测试（单元测试/集成测试/回归测试）
│   │   ├── __init__.py
│   │   ├── test_model_loader.py
│   │   ├── test_runtime_api.py
│   │   ├── test_hooks.py
│   │   ├── test_probe_csv_schema.py
│   │   ├── test_ui_routes.py
│   │   └── ...其他测试
│   │
│   ├── 【辅助模块】
│   ├── pipeline.py                  # 处理管道
│   ├── video.py                     # 视频生成
│   └── ...其他工具模块
│
└── .gitignore
```

### 五层架构说明

### 目录放置约束（v1.10）
- `src/main.py` 是唯一应用入口文件位置。
- `src/hooks.py` 是唯一模型操作层文件位置。
- `probe/` 根目录不保留 `.py` 文件；如历史遗留了 `main.py`、`hooks.py`，应删除并仅保留 `src/` 下实现。

**项目采用五层分层架构，从下到上依次为**：

#### 🏗️ 第1层：底层框架 API（Transformers Runtime）
- **职责**：加载、缓存和运行大语言模型，提供常驻模型 API
- **外部依赖**：`transformers`, `torch`, `accelerate`
- **主要文件**：`src/model_loader.py`, `src/runtime_api.py`
- **功能**：
  - 加载模型和 tokenizer 到 GPU
  - 缓存已加载的 `LocalModelBundle`
  - 启动 Llama API 服务，统一接收上层 study/probe/UI 的模型调用
  - 多次 study/API 调用时优先复用当前模型
  - 判断当前模型是否兼容新的调用需求
  - 只有在必须更换模型或配置不兼容时才重新加载
  - 管理量化配置
  - 设备映射
- **生命周期原则**：模型加载到 GPU 后默认不主动关闭；study 层不负责释放模型资源，模型生命周期由第1层运行时 API 统一管理
- **启动原则**：应用启动时先启动 Llama API 并完成模型加载/常驻准备，然后同时启动第5层 Web UI server

#### 🔌 第2层：模型操作 API 层（hooks.py）
- **职责**：基于第1层常驻模型提供模型操作 API
- **主要文件**：`src/hooks.py`
- **功能**：
  - 提取隐状态（按层、按神经元）
  - 修改模型行为（跳过层、关闭头）
  - 注入隐状态
  - 提取和对比参数
  - 临时修改模型参数并在调用结束后回退
- **特点**：低级 API，通用接口，不依赖业务逻辑；默认通过第1层 API 获取并复用已加载模型
- **兼容性原则**：如果某个 hook/study 需要的模型、量化方式、dtype、设备映射或干净模型状态与当前常驻模型不兼容，则请求第1层重新加载；否则直接复用当前模型
- **参数安全原则**：任何 hook 如果会修改模型参数、buffer 或模块状态，必须先保存原始状态，并在本次调用结束后恢复；只有无法可靠恢复时，才允许标记模型为 dirty 并触发后续重新加载

#### 📊 第3层：Probe 定义层（src/probes/）
- **职责**：定义各种探针类型及其 CSV 输出格式
- **主要文件**：
  - `src/probes/base_probe.py` - Probe 基类
  - `src/probes/linear_probe.py` - 线性探针 + CSV_Linear_Probe 格式
  - `src/probes/attribute_probe.py` - 属性探针 + CSV_Attribute_Probe 格式
  - `src/probes/concept_match.py` - 概念匹配 + CSV_Concept_Match 格式
  - `src/probes/symbolic_attributes.py` - 符号属性
- **关键设计**：**CSV 输出模式由 Probe 层负责，每个 Probe 类型自定义自己的 CSV 表格格式**
- **功能**：
  - 接收隐状态和标签数据
  - 训练探针模型
  - 生成 CSV 格式输出

#### 📚 第4层：Study 研究层（src/study/）
- **职责**：具体的研究/实验脚本，调节 Probe 参数并运行分析
- **主要文件**：
  - `src/study/study_linear_probe.py` - 线性探针研究脚本
  - `src/study/study_attribute_probe.py` - 属性探针研究脚本
  - ...其他 study 脚本
- **功能**：
  - 配置 Probe 参数
  - 加载研究数据和标注数据
  - 运行 Probe
  - 生成报告

#### 🖥️ 第5层：UI 层（src/ui/）
- **职责**：提供本地网页 server 和可视化操作页面，让用户通过浏览器调用第4层 study 和第3层 Probe
- **主要文件**：
  - `src/ui/server.py` - 启动本地 server
  - `src/ui/routes.py` - 页面路由、执行 study/probe 的 HTTP API
  - `src/ui/registry.py` - 当前可用 Study/Probe 按钮注册表
  - `src/ui/forms.py` - 每个 Study/Probe 的参数表单 schema
  - `src/ui/result_render.py` - CSV、图表和运行日志渲染
  - `src/ui/templates/` - 页面模板
  - `src/ui/static/` - 前端 CSS/JS
- **功能**：
  - 输入本地网址后显示操作页面，例如 `http://127.0.0.1:8000`
  - 页面左侧显示已有 Study/Probe 按钮
  - 页面中部显示参数输入表单
  - 点击“执行 Study”后调用对应的第4层 study，必要时进一步调用第3层 Probe
  - 页面右侧显示执行结果，默认优先显示 CSV 表格
  - 结果区提供可选图表按钮，例如彩色图、直方图等
  - 点击图表按钮后，在页面下方追加生成新图表；页面允许滚动查看历史结果
- **边界原则**：UI 层只负责交互、参数收集、任务调用和结果展示，不直接实现 Probe 训练逻辑，也不直接拼装 Probe 的 CSV 字段

### 模块分类（按文件类型）

#### 📍 入口层
- **src/main.py**: 应用主入口，同时启动 Llama API 和 Web UI server
- **src/hooks.py**: 模型操作层（第2层）

#### ⚙️ 配置层
- **src/config.py**: 配置加载和验证

#### 🎯 核心业务层
- **src/model_loader.py**: 模型加载（第1层）
- **src/extract_hidden.py**: 隐状态提取
- **src/utils.py**: 通用工具函数

#### 📊 Probe 层（第3层）
- **src/probes/**: Probe 定义
  - 每个 Probe 类型定义自己的 CSV 格式

#### 📚 Study 层（第4层）
- **src/study/**: 研究/实验脚本
  - 支持参数调节

#### 🖥️ UI 层（第5层）
- **src/ui/**: 本地网页 server 和交互页面
  - 通过按钮、表单和结果区域调用 study/probes

#### ✅ Test 目录（真正的软件测试）
- **src/test/**: 单元测试、集成测试、回归测试
  - 验证第1-5层代码是否符合设计文档

#### 🛠️ 辅助层
- **src/pipeline.py**: 业务流程协调
- **src/video.py**: 视频生成

---

## 核心模块

### 1️⃣ model_loader.py - 模型运行时 API

**职责**: 统一加载、缓存、复用模型和 tokenizer

| 接口 | 输入 | 输出 | 说明 |
|------|------|------|------|
| `get_model_bundle(config, force_reload=False)` | Config dict, bool | LocalModelBundle | 获取当前可用模型；兼容则复用，不兼容或强制时重载 |
| `load_local_model(config)` | Config dict | LocalModelBundle | 底层加载函数，只在运行时 API 判断需要加载时调用 |
| `is_model_compatible(current, config)` | LocalModelBundle, Config dict | bool | 判断当前常驻模型是否满足新调用需求 |
| `release_model_bundle()` | None | None | 显式释放模型资源；默认不由 test 自动调用 |
| `_build_quantization_config()` | dtype等参数 | BitsAndBytesConfig | 构建量化配置 |

**关键特性**:
- 支持本地模型路径和 HuggingFace Hub
- 支持 4-bit/8-bit 量化
- 自动设备映射
- 返回 `LocalModelBundle(tokenizer, model)` 对象
- 模型加载到 GPU 后默认常驻，不因单次 study 结束而关闭
- 多次 study 调用共享同一个兼容模型实例，避免重复加载
- 当模型路径、量化配置、dtype、device_map、tokenizer 或干净模型状态要求不兼容时，才重新加载

**使用示例**:
```python
from src.model_loader import get_model_bundle

config = {"model": {"model_name_or_path": "..."}}
bundle = get_model_bundle(config)
# bundle.model 和 bundle.tokenizer 可用

# 多次调用同一配置时复用 GPU 上已加载的模型
bundle2 = get_model_bundle(config)
assert bundle2 is bundle
```

**模型复用与重载规则**:
- 如果当前没有常驻模型，则加载模型到 GPU 并缓存
- 如果当前常驻模型与新调用配置兼容，则直接返回当前 `LocalModelBundle`
- 如果新调用要求不同模型、不同量化方式、不同 dtype、不同 device_map 或不同 tokenizer，则重新加载
- 如果某个 hook/study 临时修改了模型参数，必须在调用结束后回退到修改前状态
- 如果某个 hook/study 对模型结构、参数或 buffer 做了无法完全撤销的修改，后续需要干净模型时必须重新加载
- `force_reload=True` 时无条件重新加载
- 除非显式调用 `release_model_bundle()`，否则 study 结束后不自动卸载模型

---

### 2️⃣ runtime_api.py - Llama API 服务层

**职责**: 在应用启动时创建 Llama API，统一管理常驻模型并向上层提供调用入口

| 接口 | 输入 | 输出 | 说明 |
|------|------|------|------|
| `start_llama_api(config)` | Config dict | LlamaRuntimeAPI | 启动 Llama API，并确保模型加载到 GPU |
| `get_runtime_api()` | None | LlamaRuntimeAPI | 获取当前已启动的 Llama API 实例 |
| `execute_model_call(request)` | RuntimeRequest | RuntimeResult | 执行一次模型相关调用，复用常驻模型 |
| `shutdown_llama_api()` | None | None | 显式关闭 Llama API 和释放模型资源 |

**关键特性**:
- Llama API 是本地进程内 API，不要求先做公网服务
- 应用启动时先启动 Llama API，再启动网页 server
- Llama API 内部使用 `get_model_bundle()` 获取或复用常驻模型
- 第4层 study、第3层 Probe 和第5层 UI 都应通过该 API 间接使用模型
- 如果调用需要干净模型或不同模型配置，Llama API 根据运行时规则决定复用或重载

**启动示例**:
```python
from src.config import load_config
from src.runtime_api import start_llama_api

config = load_config()
llama_api = start_llama_api(config)
```

---

### 3️⃣ extract_hidden.py - 隐状态提取器

**职责**: 从模型各层提取隐藏状态向量

**三种提取模式**:

| 模式 | 函数 | 调用方式 | 返回值 |
|------|------|---------|--------|
| **单词模式** | `extract_single_word_states()` | 逐词单独 forward | 各词隐状向量表 |
| **批量模式** | `extract_all_words_state()` | 词拼接后一次 forward，取最后词 | 最后词隐状向量 |
| **序列模式** | `extract_sequence_positional_states()` | 词拼接后一次 forward，取各词位置 | 各词隐状向量表 |

**核心调用**:
```python
with torch.no_grad():
    outputs = model(**inputs, output_hidden_states=True)
    hidden = outputs.hidden_states[target_layer]  # [batch, seq_len, hidden_dim]
```

**参数**:
- `bundle`: LocalModelBundle 对象
- `words`: 词列表
- `target_layer`: 目标层编号（0-31）

---

### 4️⃣ pipeline.py - 处理管道

**职责**: 协调整个分析流程

**主要类**:
```python
class ProbePipeline:
    def __init__(config)        # 初始化时通过模型运行时 API 获取模型
    def run_single_batch()      # 单词逐个分析
    def run_multi_batch(size)   # 小批量分析
    def run_global_analysis()   # 全局分析
```

**工作流**:
```
用户输入 → 加载配置 → 获取/复用常驻模型 API → 提取隐状态 → 概念匹配 → 训练探针 → 输出结果
```

---

### 5️⃣ probes/ - Probe 定义层（第3层）

**职责**: 定义各种探针类型，每个 Probe 类型自定义 CSV 输出格式

**关键设计原则**：
- 项目默认数据输出模式为 **CSV**
- CSV 的格式定义只放在第3层 `src/probes/`
- 每个 Probe 类型独立定义自己的 CSV 表格格式
- Study 层和 pipeline 只负责调用 Probe，不直接拼装 CSV 字段
- 表格字段应清晰、自解释，字段顺序由对应 Probe 固定

**CSV 输出模式约定**：
- 每一种 Probe 对应一种 CSV 表格格式
- 每个 CSV 文件必须包含表头
- 每一行代表一个可分析记录，例如一个词在某一层上的预测、匹配或属性结果
- 不同 Probe 的字段可以不同，但同一个 Probe 的字段必须稳定
- 若需要 JSON 摘要，只能作为辅助产物；研究数据的主输出以 CSV 为准

**Probe 类型及 CSV 格式**：

#### Linear Probe（线性探针）
**文件**: `src/probes/linear_probe.py`

**CSV 输出格式**: `CSV_Linear_Probe`
```
word, layer, predicted_label, confidence, true_label, correct, feature_importance_top_3
apple, 8, color, 0.95, color, true, [dim_42: 0.8, dim_156: 0.6, dim_203: 0.4]
orange, 8, color, 0.87, color, true, [dim_42: 0.75, dim_156: 0.55, dim_203: 0.35]
...
```

**字段说明**:
- `word`: 输入词
- `layer`: 分析层数
- `predicted_label`: 模型预测的标签
- `confidence`: 预测置信度 (0-1)
- `true_label`: 真实标签
- `correct`: 是否预测正确 (true/false)
- `feature_importance_top_3`: 重要特征维度

#### Attribute Probe（属性探针）
**文件**: `src/probes/attribute_probe.py`

**CSV 输出格式**: `CSV_Attribute_Probe`
```
word, attribute_type, predicted_value, confidence, true_value, correct
apple, color, red, 0.92, red, true
apple, shape, round, 0.85, round, true
banana, color, yellow, 0.88, yellow, true
banana, shape, curved, 0.78, curved, true
...
```

**字段说明**:
- `word`: 输入词
- `attribute_type`: 属性类型（color, shape, size等）
- `predicted_value`: 预测的属性值
- `confidence`: 置信度 (0-1)
- `true_value`: 真实属性值
- `correct`: 是否预测正确

#### Concept Match（概念匹配）
**文件**: `src/probes/concept_match.py`

**CSV 输出格式**: `CSV_Concept_Match`
```
word, layer, concept, match_score, rank, semantic_similarity
apple, 8, fruit, 0.95, 1, 0.98
apple, 8, color, 0.72, 2, 0.65
apple, 8, round_shape, 0.68, 3, 0.61
...
```

**字段说明**:
- `word`: 输入词
- `layer`: 分析层数
- `concept`: 概念名称
- `match_score`: 匹配分数 (0-1)
- `rank`: 排名
- `semantic_similarity`: 语义相似度

### 6️⃣ study/ - Study 研究层（第4层）

**职责**: 具体的研究/实验脚本，调节 Probe 参数并运行

**关键特性**：
- 可调节各 Probe 的参数
- 支持参数扫描（grid search）
- 生成带参数的 CSV 输出
- 支持对比分析

**Study 文件结构**：

#### study_linear_probe.py
```python
# 参数配置
config = {
    "test_size": 0.2,
    "max_iter": 1000,
    "C": 1.0,  # 正则化参数
    "solver": "lbfgs"
}

# 运行 study
results = study_linear_probe(data, labels, config)

# 输出 CSV
save_csv("data/outputs/study_linear_probe_results.csv", results)
```

#### study_attribute_probe.py
```python
# 参数配置
config = {
    "attribute_types": ["color", "shape", "size"],
    "threshold": 0.5,
    "use_ensemble": True
}

# 运行 study
results = study_attribute_probe(data, labels, config)

# 输出 CSV
save_csv("data/outputs/study_attribute_probe_results.csv", results)
```

---

### 7️⃣ ui/ - 本地网页 UI 层（第5层）

**职责**: 提供本地 Web server 和浏览器页面，用按钮、参数表单和结果区域调用后端 study/probe

**关键设计原则**：
- UI 层是独立模块，目录固定为 `src/ui/`
- UI 层只能通过公开 API 调用第4层 `src/study/` 和第3层 `src/probes/`
- UI 层的请求入口是网页 server，但底层模型调用必须走已启动的 Llama API
- UI 层不直接实现 Probe 算法，不直接修改模型，不直接拼装 Probe CSV 字段
- UI 的每个按钮必须对应一个注册项，注册项声明它调用的 study/probe、参数 schema、输出类型
- 页面默认优先展示 CSV；图表是基于 CSV 或结果数据的附加展示
- server 默认只绑定 `127.0.0.1`，作为本地研究工具使用

**页面布局**：
- 左侧：Study/Probe 按钮区，初始放入已有 `study_linear_probe.py`、`study_attribute_probe.py`、`linear_probe.py`、`attribute_probe.py`、`concept_match.py`
- 中部：参数输入表单，根据当前按钮的参数 schema 动态显示
- 右侧：执行结果区，点击“执行 Study”后显示 CSV 表格、运行状态和错误信息
- 下方：可滚动图表区，点击“生成彩色图”“生成直方图”等按钮后追加新图表

**UI 执行流程**：
```
浏览器页面
  → 选择 Study/Probe 按钮
  → 填写参数表单
  → 点击执行 Study
  → routes.py 接收请求
  → registry.py 找到对应调用项
  → 调用 src/study/ 或 src/probes/
  → study/probes 通过 Llama API 调用常驻模型
  → 返回 CSV 路径 / CSV rows / metrics
  → result_render.py 渲染 CSV 表格
  → 用户可继续点击图表按钮生成附加图
```

**按钮注册表示例**：
```python
UI_ACTIONS = {
    "study_linear_probe": {
        "label": "Linear Probe Study",
        "type": "study",
        "target": "src.study.study_linear_probe",
        "form_schema": "linear_probe_form",
        "default_output": "csv"
    },
    "study_attribute_probe": {
        "label": "Attribute Probe Study",
        "type": "study",
        "target": "src.study.study_attribute_probe",
        "form_schema": "attribute_probe_form",
        "default_output": "csv"
    },
    "concept_match": {
        "label": "Concept Match",
        "type": "probe",
        "target": "src.probes.concept_match",
        "form_schema": "concept_match_form",
        "default_output": "csv"
    }
}
```

**结果展示规则**：
- 执行完成后，右侧结果区默认显示 CSV 表格
- CSV 的字段和顺序以第3层 Probe 定义为准
- 如果结果文件较大，UI 可分页或只预览前 N 行
- 图表按钮不覆盖 CSV 主结果，只在页面下方追加展示
- 图表生成应基于已完成的 CSV/result 数据，不重复执行耗时 study，除非用户重新点击执行
- 每次执行 study/probe 应记录运行参数、输出文件路径、状态和错误信息

---

### 8️⃣ test/ - 软件测试目录（真正 test 含义）

**职责**: 存放真正的软件测试代码，用于验证项目实现是否符合本文档设计

**关键设计原则**：
- `src/test/` 只表示真正的软件测试，不放研究/实验脚本
- 原第4层研究脚本统一放入 `src/study/`
- 软件测试应覆盖模型运行时 API、钩子回退、Probe CSV schema、UI routes 等关键行为
- 测试代码不得依赖必须加载完整大模型的路径；需要时使用 mock、stub 或小模型

**测试文件示例**：
- `src/test/test_model_loader.py` - 验证模型加载、缓存、重载策略
- `src/test/test_runtime_api.py` - 验证 Llama API 启动和调用分发
- `src/test/test_hooks.py` - 验证钩子移除、参数备份和回退
- `src/test/test_probe_csv_schema.py` - 验证每个 Probe 的 CSV 字段稳定
- `src/test/test_ui_routes.py` - 验证 UI action 路由到 study/probe

---

### 9️⃣ hooks.py - 模型操作层（第2层）

**职责**: 提供基于常驻模型的直接操作 API

**主要功能**:
1. **隐状态提取**: `extract_layer_hidden_states()` - 提取某个/多个/所有层的隐状态
2. **层操作**: `skip_layers()` - 跳过指定层
3. **头操作**: `disable_attention_heads()` - 关闭某些注意力头
4. **FFN操作**: `extract_ffn_neuron_hidden_states()` - 提取 FFN 神经元激活
5. **参数操作**: `extract_neuron_parameters()` - 提取参数、`compare_neuron_parameters()` - 对比参数

**API 模式要求**:
- hooks 默认不负责加载或关闭模型，只通过 `LocalModelBundle` 或第1层运行时 API 使用当前常驻模型
- 临时 hook 必须在单次调用结束后移除，避免污染后续 study
- 如果 hook 会临时修改模型参数、buffer 或模块状态，必须先备份原值，并使用 `try/finally` 或上下文管理器保证调用结束后恢复
- 参数恢复必须发生在同一次 hook/study 调用结束时，不能依赖后续 study 手动清理
- 如果参数恢复失败，或操作会永久改变模型结构、参数或 forward 行为，必须标记当前模型状态为 dirty
- 当后续 study 需要干净模型且当前模型为 dirty 时，第1层必须重新加载模型
- 不同 hook/study 如果能共享同一个模型配置，应通过 API 复用当前 GPU 模型

**参数修改回退规范**:
- 修改前必须记录被修改参数的原始值，记录范围只覆盖本次 hook 实际修改的参数
- 修改可以作用于 `nn.Parameter`、buffer 或模块属性，但恢复时必须回到调用前状态
- 推荐用上下文管理器封装参数修改，退出上下文时自动恢复
- 恢复后应移除所有 hook handle，避免后续 forward 继续触发旧逻辑
- 对大参数只在必要时备份，优先记录局部 slice；如果必须整块备份，应评估 GPU/CPU 内存开销
- 如果为了节省显存把备份放到 CPU，恢复时必须放回原参数所在 device 和 dtype
- 如果无法证明参数已经恢复干净，必须将常驻模型标记为 dirty

**使用示例**:
```python
from hooks import extract_layer_hidden_states, disable_attention_heads

# 提取隐状态
states = extract_layer_hidden_states(model, layer_indices=[5, 10], input_ids=input_ids)

# 关闭头
disabler = disable_attention_heads(model, head_indices=[0, 1], layer_indices=[5])
# 执行推理...
disabler.remove_all_hooks()
```

**临时参数修改示例**:
```python
from hooks import temporary_parameter_patch

with temporary_parameter_patch(model, patches):
    # 在上下文内部运行修改后的模型
    outputs = model(**inputs)

# 退出上下文后，参数必须已经恢复到调用前状态
```

---

## 数据流

### 应用启动流程

```
启动项目 (src/main.py，无参数)
    ↓
[1] 加载配置
    ├─ 无参数时读取 configs/custom.yaml
    └─ 用户显式指定其他配置文件时覆盖
    ↓
[2] 启动 Llama API (src/runtime_api.py)
    ├─ 调用 model_loader.py::get_model_bundle
    ├─ 将 Llama 模型加载到 GPU
    ├─ 建立常驻模型 API
    └─ 等待 study/probe/UI 调用
    ↓
[3] 启动 Web UI server (src/ui/server.py)
    ├─ 绑定本地地址，例如 http://127.0.0.1:8000
    ├─ 加载 UI action 注册表
    └─ 提供网页和执行 API
    ↓
[4] 两个服务同时保持运行
    ├─ Llama API：负责模型常驻、复用、必要重载
    └─ Web UI server：负责页面、按钮、表单和结果展示
```

**启动原则**:
- `src/main.py` 是统一启动入口
- 启动时必须同时启动两个服务：Llama API 和 Web UI server
- Llama API 负责底层模型运行，不负责页面展示
- Web UI server 负责网页交互，不直接加载或关闭模型
- 用户在网页点击某个 study 按钮后，Web UI server 才调用对应的第4层 study 或第3层 Probe 代码

### 完整数据流

```
用户输入 (CLI 或 UI 参数表单)
    ↓
[1] 加载配置
    ├─ 无参数时读取 configs/custom.yaml
    └─ 返回: Config dict
    ↓
[2] 获取模型运行时 API (model_loader.py::get_model_bundle)
    ├─ 若 GPU 上无常驻模型：加载 tokenizer 和模型（可选量化）
    ├─ 若已有兼容模型：直接复用
    ├─ 若模型/配置/状态不兼容：重新加载
    └─ 返回: LocalModelBundle(tokenizer, model)
    ↓
[3] 预处理词 (extract_hidden.py)
    ├─ Tokenize 词
    ├─ 转移到 GPU
    └─ 返回: tensor inputs
    ↓
[4] 提取隐状态 (三种模式，见下表)
    ├─ 执行 model forward pass
    ├─ 提取 hidden_states[target_layer]
    └─ 返回: hidden vectors
    ↓
[5] 概念匹配 (concept_match.py)
    ├─ 加载概念目录
    ├─ 计算相似度
    └─ 返回: concept scores
    ↓
[6] 运行 Probe（src/probes/）
    ├─ 加载标注数据
    ├─ 训练 LogisticRegression
    └─ 返回: probe model + metrics + CSV rows
    ↓
[7] 清理临时模型操作
    ├─ 移除临时 hooks
    ├─ 回退临时参数修改
    └─ 若无法恢复干净：标记常驻模型 dirty
    ↓
[8] 输出结果
    ├─ CSV 主输出（由对应 Probe 定义表格格式）
    ├─ JSON 摘要（可选辅助）
    ├─ 可视化图表
    └─ MP4 视频 (可选)
```

### UI 交互数据流（第5层）

```
用户打开本地网址
    ↓
[1] src/ui/server.py 启动本地 server
    └─ 默认地址: http://127.0.0.1:8000
    ↓
[2] 页面加载 Study/Probe 注册表
    ├─ src/ui/registry.py
    ├─ 显示已有 study 按钮
    └─ 显示已有 Probe 按钮
    ↓
[3] 用户选择按钮并填写参数 form
    ├─ 参数 schema 来自 src/ui/forms.py
    └─ 参数可以覆盖 configs/custom.yaml 的相关字段
    ↓
[4] 点击执行 Study
    ├─ src/ui/routes.py 接收请求
    ├─ 校验参数
    └─ 调用对应 src/study/ 或 src/probes/
    ↓
[5] 后端执行分析
    ├─ 通过已启动的 Llama API 获取/复用常驻模型
    ├─ 如需钩子则调用第2层 hooks.py
    ├─ 调用第3层 Probe
    └─ 由第4层 study 协调参数和执行
    ↓
[6] 返回结果到 UI
    ├─ CSV 文件路径或 CSV rows
    ├─ metrics / logs / error message
    └─ 可选图表数据
    ↓
[7] 页面右侧显示 CSV
    ├─ 默认显示 CSV 表格
    └─ 保留运行参数和输出路径
    ↓
[8] 用户点击图表按钮
    ├─ 基于已有 CSV/result 数据生成图表
    └─ 在页面下方追加图表，页面可滚动查看
```

### 隐状态提取三种模式对比

```
输入: ["apple", "orange", "banana"]
目标层: 8

┌─────────────────────────────────────────────┐
│ 模式 1: 单词模式                            │
│ 逐词单独 forward pass                       │
│                                              │
│ Forward 1: "apple"    → [1, seq_len, 4096] │
│ Forward 2: "orange"   → [1, seq_len, 4096] │
│ Forward 3: "banana"   → [1, seq_len, 4096] │
│                                              │
│ 返回: [[4096], [4096], [4096]]              │
│ 特点: 最准确，但最慢                        │
└─────────────────────────────────────────────┘

┌─────────────────────────────────────────────┐
│ 模式 2: 批量模式                            │
│ 词拼接后一次 forward，取最后词              │
│                                              │
│ 输入: "apple orange banana"                  │
│ Forward: 一次 forward pass                  │
│ 提取: hidden_states[8][-1] (最后位置)       │
│                                              │
│ 返回: [4096]  (只有最后词)                  │
│ 特点: 最快，但受上下文影响                  │
└─────────────────────────────────────────────┘

┌─────────────────────────────────────────────┐
│ 模式 3: 序列模式                            │
│ 词拼接后一次 forward，取各词自身位置        │
│                                              │
│ 输入: "apple orange banana"                  │
│ Forward: 一次 forward pass                  │
│ 提取: 各词自身位置的隐状态                  │
│ hidden_states[8][word_positions]            │
│                                              │
│ 返回: [[4096], [4096], [4096]]              │
│ 特点: 折中，兼顾速度和准确性                │
└─────────────────────────────────────────────┘
```

---

## 配置系统

### 配置文件结构

**默认文件**: `configs/custom.yaml`

**命名规则**:
- 正常启动时不需要输入配置文件名，系统默认读取 `configs/custom.yaml`
- `custom.yaml` 的初始内容从原 `default.yaml` 复制而来
- 用户要改默认行为时，直接编辑 `configs/custom.yaml`
- 不再把 `default.yaml` 或临时编号文件（如 `4.yaml`）作为正常启动入口
- 如果用户确实需要实验配置，可以显式传入其他 YAML 文件名

```yaml
# 模型配置
model:
  model_name_or_path: "C:/AI_Model/Llama3_8B_Instruct"
  torch_dtype: "bfloat16"           # float16, bfloat16, float32
  device_map: "auto"
  # 量化配置
  load_in_4bit: false
  load_in_8bit: false
  bnb_4bit_compute_dtype: "bfloat16"
  bnb_4bit_use_double_quant: true
  bnb_4bit_quant_type: "nf4"

# 分析配置
analysis:
  target_layer: 8                   # 0-31
  extraction_mode: "single"         # single, batch, sequence
  # 报告详度
  top_k_dims: 24
  top_k_words_per_dim: 8
  top_k_concepts: 8
  top_k_predictions: 5

# 运行时 API 配置
runtime:
  api_mode: true
  start_llama_api_on_boot: true
  keep_model_loaded: true
  reload_policy: "auto"             # auto, always, never
  require_clean_model: false

# UI 配置
ui:
  enabled: true
  start_server_on_boot: true
  host: "127.0.0.1"
  port: 8000
  open_browser: false
  default_view: "study_linear_probe"
  max_csv_preview_rows: 200
  allow_chart_generation: true

# 探针训练配置
probe:
  linear_probe:
    test_size: 0.2
    max_iter: 1000
    C: 1.0
    solver: "lbfgs"

# 数据配置
data:
  word_file: "data/words.txt"
  label_file: "data/labels.csv"
  concept_catalog: "data/concept_catalog.yaml"

# 输出配置
output:
  save_dir: "data/outputs/"
  mode: "CSV"
  save_json: false
  save_csv: true
  generate_video: false
```

### 配置优先级

1. **默认值** (代码内)
2. **默认配置文件** (`configs/custom.yaml`)
3. **用户显式指定的配置文件** (可选 YAML)
4. **UI 表单参数** (网页执行时覆盖本次 study/probe 调用)

### 启动配置约定

- 正常启动命令不带任何配置参数
- `src/main.py` 默认调用 `load_config()`，而 `load_config()` 默认读取 `configs/custom.yaml`
- 如果用户传入其他配置文件名，则只覆盖本次启动使用的配置来源
- 文档、示例和 UI 默认都以 `configs/custom.yaml` 为基准

---

## API 接口

### 公共 API (面向用户/研究脚本)

#### 模型加载
```python
from src.model_loader import get_model_bundle
from src.config import load_config

config = load_config()
bundle = get_model_bundle(config)
# bundle.model: AutoModelForCausalLM
# bundle.tokenizer: AutoTokenizer
```

#### Llama API 启动
```python
from src.config import load_config
from src.runtime_api import start_llama_api, get_runtime_api

config = load_config()
llama_api = start_llama_api(config)

# 后续 study/probe/UI 调用复用同一个 Llama API
llama_api = get_runtime_api()
```

#### 隐状态提取
```python
from src.extract_hidden import (
    extract_single_word_states,
    extract_all_words_state,
    extract_sequence_positional_states
)

# 单词模式
states = extract_single_word_states(bundle, ["apple", "orange"], target_layer=8)

# 批量模式
state = extract_all_words_state(bundle, ["apple", "orange"], target_layer=8)

# 序列模式
states = extract_sequence_positional_states(bundle, ["apple", "orange"], target_layer=8)
```

#### 模型操作 (hooks.py)
```python
from hooks import (
    extract_layer_hidden_states,
    disable_attention_heads,
    skip_layers,
    extract_ffn_neuron_hidden_states,
    extract_neuron_parameters,
    compare_neuron_parameters
)
from src.config import load_config
from src.model_loader import get_model_bundle

config = load_config()
bundle = get_model_bundle(config)

# 提取隐状态
states = extract_layer_hidden_states(bundle.model, layer_indices=[5, 10], input_ids=input_ids)

# 关闭头
disabler = disable_attention_heads(bundle.model, head_indices=0, layer_indices=None)

# 跳过层
output = skip_layers(bundle.model, layer_indices=[3, 4], input_ids=input_ids)

# 提取 FFN 神经元
ffn_states = extract_ffn_neuron_hidden_states(bundle.model, layer_indices=5, input_ids=input_ids)

# 提取参数
params = extract_neuron_parameters(bundle.model, layer_idx=5, param_type='weight')

# 对比参数变化
result = compare_neuron_parameters(bundle.model, layer_idx=5,
                                  before_fn=func1, after_fn=func2)
```

#### 探针训练
```python
from src.probes.linear_probe import train_linear_probe, evaluate_probe

probe, metrics = train_linear_probe(X_train, y_train, config)
accuracy = evaluate_probe(probe, X_test, y_test)
```

#### 概念匹配
```python
from src.probes.concept_match import match_concepts

scores = match_concepts(hidden_states, concept_catalog)
```

#### 本地 UI Server（第5层）
```python
from src.config import load_config
from src.runtime_api import start_llama_api
from src.ui.server import run_ui_server

config = load_config()
start_llama_api(config)
run_ui_server(config)
# 浏览器访问: http://127.0.0.1:8000
```

#### 应用双服务启动
```python
from src.main import start_app

start_app()
# 同时启动:
# - Llama API: 常驻模型和底层调用
# - Web UI server: 本地网页和按钮执行入口

# 可选：用户显式指定其他配置文件
start_app("configs/my_experiment.yaml")
```

#### UI 执行接口
```python
from src.ui.registry import get_ui_action
from src.ui.routes import execute_ui_action

action = get_ui_action("study_linear_probe")
result = execute_ui_action(action_id="study_linear_probe", params={
    "target_layer": 8,
    "test_size": 0.2,
    "max_iter": 1000
})

# result 至少包含:
# - csv_path 或 csv_rows
# - metrics
# - status
# - error_message
```

---

## 模块依赖图

### 整体架构依赖

```
【第5层】UI 层 - src/ui/
├─→ server.py (本地网页 server)
├─→ routes.py (页面路由和执行 API)
├─→ registry.py (Study/Probe 按钮注册表)
├─→ forms.py (参数表单 schema)
└─→ result_render.py (CSV/图表展示)
     ↓
【第4层】Study 研究层 - src/study/
├─→ study_linear_probe.py
├─→ study_attribute_probe.py
└─→ ...study 脚本
     ↓
【第3层】Probe 层 - src/probes/
├─→ base_probe.py
├─→ linear_probe.py
├─→ attribute_probe.py
└─→ concept_match.py
     ↓
【第2层】模型操作 API 层
├─→ src/hooks.py (通过 API 操作常驻模型)
├─→ src/extract_hidden.py (提取隐状态)
├─→ src/runtime_api.py (Llama API 调用入口)
└─→ src/model_loader.py (获取/复用/必要时重载模型)
     ↓
【第1层】底层框架 API
└─→ transformers (Pytorch, HuggingFace)
```

### 详细依赖关系

```
ui/ (UI 层，第5层)
  ├─→ server.py (启动本地 server)
  ├─→ routes.py (HTTP API)
  ├─→ registry.py (Study/Probe 操作注册)
  ├─→ forms.py (参数表单)
  ├─→ result_render.py (CSV/图表结果渲染)
  │
  ├─→ study/ (调用 Study 层，第4层)
  │   ├─→ study_linear_probe.py
  │   └─→ study_attribute_probe.py
  │
  └─→ probes/ (必要时直接调用 Probe 层，第3层)
      ├─→ linear_probe.py
      ├─→ attribute_probe.py
      └─→ concept_match.py

src/main.py (入口)
  ├─→ config.py (配置加载)
  ├─→ runtime_api.py (启动 Llama API，第1层)
  │   └─→ model_loader.py (加载/复用模型)
  │       └─→ transformers.AutoTokenizer
  │       └─→ transformers.AutoModelForCausalLM
  │
  ├─→ ui/server.py (启动 Web UI server，第5层)
  │   └─→ routes.py
  │       └─→ study/ 或 probes/
  │
  └─→ pipeline.py (业务流程)
      ├─→ extract_hidden.py (提取隐状态)
      │   └─→ model forward pass
      │
      ├─→ probes/ (Probe 层，第3层)
      │   ├─→ linear_probe.py
      │   ├─→ attribute_probe.py
      │   └─→ concept_match.py
      │
      └─→ video.py (可视化)

study/ (Study 研究层，第4层)
  ├─→ study_linear_probe.py
  ├─→ study_attribute_probe.py
  │
  ├─→ probes/ (调用 Probe 类)
  │   └─→ 生成 CSV 格式输出
  │
  └─→ hooks.py (通用模型操作 API)
      ├─→ extract_layer_hidden_states()
      ├─→ disable_attention_heads()
      ├─→ skip_layers()
      └─→ ...
```

---

## 扩展机制

### 1️⃣ 添加新的分析脚本

**步骤**:
1. 在 `研究脚本/` 下创建新文件
2. 导入必要的模块
3. 使用现有 API 进行分析

**模板**:
```python
# 研究脚本/my_analysis.py
from src.model_loader import get_model_bundle
from src.config import load_config
from hooks import extract_layer_hidden_states

config = load_config()
bundle = get_model_bundle(config)

# 执行分析...
```

### 2️⃣ 添加新的探针类型

**步骤**:
1. 在 `src/probes/` 下创建新文件 `new_probe.py`
2. 实现 Probe 类或函数
3. 在新 Probe 中定义自己的 `CSV_New_Probe` 表格格式
4. 在 `pipeline.py` 中添加调用

**接口规范**:
```python
def train_new_probe(X_train, y_train, config):
    """
    Args:
        X_train: (n_samples, n_features) 隐状态矩阵
        y_train: (n_samples,) 标签
        config: 配置字典

    Returns:
        probe: 训练好的探针模型
        metrics: 评估指标字典
        csv_rows: 符合 CSV_New_Probe 格式的表格行
    """
    pass
```

### 3️⃣ 扩展 hooks.py

**步骤**:
1. 编辑 `hooks.py`
2. 添加新的钩子函数或类
3. 添加文档说明
4. 明确该 hook 是否会污染常驻模型状态；如果会，必须触发 dirty 标记或要求重新加载
5. 如果 hook 会修改参数，必须同时实现参数备份和回退逻辑

**函数命名规范**:
- 提取操作: `extract_*`
- 修改操作: `modify_*` 或 `disable_*`
- 查询操作: `get_*`

### 4️⃣ 添加新的 UI 按钮或页面入口

**步骤**:
1. 在 `src/ui/registry.py` 中添加新的 action 注册项
2. 在 `src/ui/forms.py` 中定义该 action 的参数表单 schema
3. 将 action 绑定到已有 `src/study/` 研究函数或 `src/probes/` Probe 函数
4. 确认返回结果包含 CSV 文件路径或 CSV rows
5. 如需图表，在 `src/ui/result_render.py` 中添加基于结果数据的图表生成函数

**接口规范**:
```python
{
    "id": "study_linear_probe",
    "label": "Linear Probe Study",
    "type": "study",
    "target": "src.study.study_linear_probe",
    "form_schema": "linear_probe_form",
    "default_output": "csv",
    "chart_actions": ["histogram", "color_map"]
}
```

**UI 扩展原则**:
- 新按钮不应绕过第4层 study 或第3层 Probe 直接操作模型
- 参数表单只负责收集参数，参数含义必须由对应 study/probe 定义
- 图表按钮必须基于已生成结果，不改变原始 CSV 输出格式
- 删除按钮时，只从 UI 注册表移除，不删除底层 study/probe 文件

### 5️⃣ 添加真正的软件测试

**步骤**:
1. 在 `src/test/` 下创建 `test_*.py`
2. 只测试代码行为和接口契约，不把研究实验脚本放入这里
3. 优先使用 mock、stub 或小模型，避免单元测试强依赖完整 Llama 模型
4. 覆盖关键边界：模型复用/重载、钩子参数回退、CSV schema、UI action 路由

**示例**:
```python
# src/test/test_probe_csv_schema.py
def test_linear_probe_csv_columns_are_stable():
    expected = [
        "word",
        "layer",
        "predicted_label",
        "confidence",
        "true_label",
        "correct",
        "feature_importance_top_3",
    ]
    assert get_linear_probe_csv_columns() == expected
```

---

## 修改流程

### 如何修改项目

**遵循这个流程，可以最小化工作量**:

1. **修改设计文档**
   ```
   编辑本文档，描述你想要的改变
   - 修改某个模块的职责
   - 添加新的 API
   - 改变数据流
   ```

2. **审视影响范围**
   ```
   根据模块依赖图，识别哪些模块会受影响
   - 直接依赖
   - 间接依赖
   - 副作用
   ```

3. **修改代码**
   ```
   只修改受影响的模块
   - 更新函数签名
   - 修改实现
   - 更新调用方
   ```

4. **更新相关文档**
   ```
   - HOOKS_USAGE.md (如果涉及 hooks.py)
   - API 文档
   - 示例代码
   ```

### 修改示例

#### 示例 1: 添加新的隐状态提取模式

**步骤 1**: 修改设计文档
```
在"隐状态提取三种模式对比"中添加新模式
更新 extract_hidden.py 的职责描述
```

**步骤 2**: 修改代码
```python
# src/extract_hidden.py
def extract_windowed_states(bundle, words, target_layer, window_size=3):
    """新的窗口模式提取"""
    pass
```

**步骤 3**: 更新 pipeline.py
```python
# src/pipeline.py
if config["analysis"]["extraction_mode"] == "windowed":
    states = extract_windowed_states(...)
```

#### 示例 2: 添加新的模块操作

**步骤 1**: 修改设计文档
```
在"hooks.py"模块说明中添加新函数
更新数据流图
```

**步骤 2**: 修改代码
```python
# hooks.py
def freeze_layer(model, layer_indices):
    """冻结某些层的参数"""
    pass
```

**步骤 3**: 更新 HOOKS_USAGE.md
```markdown
## 冻结层 (Freeze Layers)

### 冻结某一层
...
```

---

## 版本历史

| 版本 | 日期 | 主要变更 |
|------|------|---------|
| 1.10 | 2026-05-24 | 明确目录规范：`src/main.py` 与 `src/hooks.py` 为唯一位置；`probe/` 根目录不保留 `.py` 文件 |
| 1.9 | 2026-05-24 | 收尾 `outputs/` → `data/outputs/` 迁移：明确剩余待改脚本与完成判定标准（先改文档再改代码） |
| 1.7 | 2026-05-24 | 将原第4层 `src/tests/` 研究脚本改名为 `src/study/`；新增 `src/test/` 作为真正的软件测试目录 |
| 1.6 | 2026-05-24 | 将默认启动配置统一为 `configs/custom.yaml`；正常启动不需要输入配置文件名，仍支持显式指定其他 YAML |
| 1.5 | 2026-05-24 | 明确应用启动时同时启动 Llama API 和 Web UI server；网页按钮点击后再调用对应 study/probe 底层代码 |
| 1.4 | 2026-05-24 | 增加第5层 UI 层：`src/ui/` 本地网页 server、Study/Probe 按钮、参数表单、CSV 结果展示和可追加图表 |
| 1.3 | 2026-05-24 | 明确 hooks 临时修改模型参数时必须备份并在调用结束后回退；无法恢复时标记 dirty 并按需重新加载 |
| 1.2 | 2026-05-24 | 将第1层和第2层调整为模型运行时 API 模式；模型加载到 GPU 后默认常驻，多次 study 优先复用，必要时才重新加载 |
| 1.1 | 2026-05-24 | 将项目主数据输出模式改为 CSV；明确第3层 probes/ 负责每个 Probe 的 CSV 表格格式 |
| 1.0 | 2026-05-24 | 初始版本，整合现有项目文档 + hooks.py 模块 |

---

## 常见问题

**Q: 我想添加一个新功能，应该从哪里开始？**
A: 先修改此设计文档，在适当位置添加功能描述，然后根据修改后的设计文档修改代码。

**Q: hooks.py 和 extract_hidden.py 有什么区别？**
A: extract_hidden.py 是高级 API，专为项目的隐状态提取业务设计。hooks.py 是低级模型操作 API，默认使用第1层已经常驻在 GPU 上的模型，可供任何研究脚本使用。

**Q: 多次运行 study 会重复加载大语言模型吗？**
A: 不会。第1层模型运行时 API 会优先复用 GPU 上已经加载的兼容模型；只有模型路径、量化配置、dtype、设备映射、tokenizer 或模型状态不兼容时才重新加载。

**Q: hooks 修改了模型参数怎么办？**
A: 必须在 hook/study 调用内部先备份原参数，运行结束后立即恢复。恢复失败或无法确认恢复干净时，必须标记模型 dirty，后续需要干净模型时由第1层重新加载。

**Q: UI 层可以直接调用模型或拼装 CSV 吗？**
A: 不可以。UI 层只负责本地网页交互、参数收集、调用 study/probe 和展示结果。模型操作必须走第1-2层 API，CSV 字段格式必须由第3层 Probe 定义。

**Q: 网页按钮点击后执行什么？**
A: 按钮通过 `src/ui/registry.py` 映射到具体的第4层 study 或第3层 Probe。用户填写 form 参数并点击执行后，UI 调用对应 Python 函数，右侧优先展示 CSV，图表按钮可基于已有结果继续生成附加图。

**Q: `src/study/` 和 `src/test/` 有什么区别？**
A: `src/study/` 放研究/实验脚本，是第4层业务流程的一部分；`src/test/` 才是真正的软件测试目录，用来做单元测试、集成测试和回归测试。

**Q: 启动项目时启动哪些服务？**
A: `src/main.py` 同时启动两个服务：Llama API 和 Web UI server。Llama API 负责常驻模型和底层调用；Web UI server 负责网页、按钮、表单和结果展示。

**Q: 正常启动时需要输入配置文件名吗？**
A: 不需要。正常启动直接运行 `main`，默认读取 `configs/custom.yaml`。如果用户要使用其他实验配置，可以显式传入另一个 YAML 文件名。

**Q: 网页点击 study 按钮时会重新启动模型吗？**
A: 不会。按钮点击后 Web UI server 调用对应 study/probe，底层通过已启动的 Llama API 复用常驻模型；只有配置或模型状态不兼容时才由 Llama API 重新加载。

**Q: 如何确保修改不会破坏现有功能？**
A: 检查模块依赖图，确保只修改受影响的模块，其他模块的接口保持不变。

**Q: 我想修改 hooks.py，需要更新这个设计文档吗？**
A: 是的。如果是重要改变，应该更新本文档的"hooks.py"部分，然后同步更新 HOOKS_USAGE.md。

---

## 联系方式

有关设计文档的问题，请参考：
- 本文件: `PROJECT_DESIGN.md`
- 模块文档: 各模块的 docstring
- 使用指南: `HOOKS_USAGE.md`, `README.md`
