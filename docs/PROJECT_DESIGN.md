# LLM Probe 项目全局设计文档

**版本**: 1.32
**最后更新**: 2026-05-24
**作用**: 项目的单一真实来源 (Single Source of Truth)
**使用说明**: 修改项目时，先修改此文档，然后根据此文档的要求修改代码

## 文档工作规则（新增）
- `PROJECT_DESIGN.md` 主要保留“文件夹结构”和“跨文件协作规则”，保持精简。
- 具体到某个 `.py` 文件的需求，不再在主设计文档重复展开。
- 每个 `.py` 文件的具体需求写在该文件开头注释中（职责、输入输出、边界）。
- 后续新增功能时，优先更新对应 `.py` 文件头注释，再同步必要的目录级规则到本文件。
- `docs/design_split/` 下的拆分文档仍是有效规范，涉及跨文件流程/全局规则时必须同步维护。
- 实际工作时按需读取 split 文档：只有遇到跨模块、跨层级问题时再打开对应 split 文件。
- 无法合理下沉到单个 `.py` 文件的内容（例如跨模块数据流、依赖关系、维护流程）保留在 split 文档中。

---

## 设计拆分索引

- [核心模块](./design_split/CORE_MODULES.md)
- [数据流](./design_split/DATA_FLOW.md)
- [配置与 API](./design_split/CONFIG_AND_API.md)
- [依赖图与扩展机制](./design_split/DEPENDENCY_AND_EXTENSION.md)
- [修改流程与 FAQ](./design_split/WORKFLOW_FAQ.md)

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
│   ├── outputs/                     # 📁 输出结果（v1.8更新：已从robe根目录迁移到此处）
│       ├── shape_words/             # 形状相关实验输出
│       ├── color_words/          # 颜色相关实验输出
│       └── ...                      # 其他实验输出
│   └── cache/                       # 全词表隐状态缓存（不纳入 Git）
│       ├── hidden_states.f16.bin    # 定长主数据文件（token_id 直索引）
│       └── hidden_states.done.bin   # 构建进度标记（断点续跑）
│
├── docs/                            # 📖 文档和依赖
│   ├── PROJECT_DESIGN.md            # 🎯 项目全局设计文档（THIS FILE）
│   ├── ARCHITECTURE.md              # 架构详细说明
│   ├── README.md                    # 快速开始
│   ├── HOOKS_USAGE.md               # src/utils/hooks.py 使用指南
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
│   ├── utils/                       # 工具模块目录
│   │   ├── hooks.py                 # 🔌 模型操作钩子
│   │   ├── extract_hidden.py        # 隐状态提取
│   │   ├── utils.py                 # 通用工具函数
│   │   ├── video.py                 # 视频生成
│   │   └── visualize_*.py           # 可视化工具
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
│   └── ...其他工具模块
│
└── .gitignore
```

### 五层架构说明

### 目录放置约束（v1.13）
- `src/main.py` 是唯一应用入口文件位置。
- `src/utils/hooks.py` 是唯一模型操作层文件位置。
- `probe/` 根目录不保留 `.py` 文件；如历史遗留了 `main.py`、`hooks.py`，应删除并仅保留 `src/` 下实现。
- `data/` 目录是本地数据与实验产物目录，不纳入 Git；应在 `.gitignore` 中忽略 `data/**`。
- 全词表隐状态缓存采用定长二进制文件，路径默认位于 `data/cache/`，不纳入 Git。

### 工具模块归位（v1.13）
- 新增目录：`src/utils/`
- 首批迁移模块（工具类）：
  - `src/hooks.py` → `src/utils/hooks.py`
  - `src/extract_hidden.py` → `src/utils/extract_hidden.py`
  - `src/utils.py` → `src/utils/utils.py`
  - `src/video.py` → `src/utils/video.py`
  - `src/visualize_single_word.py` → `src/utils/visualize_single_word.py`
  - `src/visualize_multi_word.py` → `src/utils/visualize_multi_word.py`
  - `src/visualize_color_experiment.py` → `src/utils/visualize_color_experiment.py`
- `src/` 顶层优先保留“入口与核心编排”模块（如 `main.py`、`config.py`、`model_loader.py`、`pipeline.py`）。

### 启动与路径约束（v1.12）
- 标准启动命令为 `python -m src.main`（工作目录为 `probe/` 根目录）。
- `src/main.py` 需要兼容“文件方式启动”（`python src/main.py`）：必须在导入 `from src...` 之前完成项目根路径注入。
- 配置文件路径（如 `configs/custom.yaml`）与数据路径（如 `data/...`）的相对路径统一以 `probe/` 根目录为基准解析，不依赖调用时当前目录。
- UI/Study/Probe 之间所有文件访问默认在 `probe/` 根目录语义下进行，避免因运行目录不同导致路径失效。

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

#### 🔌 第2层：模型操作 API 层（utils/hooks.py）
- **职责**：基于第1层常驻模型提供模型操作 API
- **主要文件**：`src/utils/hooks.py`
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
- **src/utils/hooks.py**: 模型操作层（第2层）

#### ⚙️ 配置层
- **src/config.py**: 配置加载和验证

#### 🎯 核心业务层
- **src/model_loader.py**: 模型加载（第1层）
- **src/utils/extract_hidden.py**: 隐状态提取
- **src/utils/utils.py**: 通用工具函数

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
- **src/utils/video.py**: 视频生成

