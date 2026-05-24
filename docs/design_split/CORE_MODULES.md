
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

### 3️⃣ utils/extract_hidden.py - 隐状态提取器

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

### 4️⃣ utils/token_hidden_store.py - 全词表隐状态定长缓存

**职责**: 将模型词表的全 token、全层隐状态写入单一二进制定长文件，并支持断点续写与随机读取。

**存储规则**:
- 主文件（按协议分文件）:
  - `data/cache/hidden_states.bos0_assistant0.f16.bin`
  - `data/cache/hidden_states.bos1_assistant0.f16.bin`
  - `data/cache/hidden_states.bos1_assistant1.f16.bin`
- 进度文件（按协议分文件）:
  - `data/cache/hidden_states.bos0_assistant0.done.bin`
  - `data/cache/hidden_states.bos1_assistant0.done.bin`
  - `data/cache/hidden_states.bos1_assistant1.done.bin`
- dtype: `float16`（每值 2 字节）
- 维度: `n_layers=36`, `hidden_dim=4096`
- token 规模: 使用 `max_token_id + 1` 作为记录总数（当前 Llama3 本地实测为 128256）

**协议规则（BOS / assistant）**:
- `bos=false` 时，`assistant` 必须为 `false`（禁止 `bos0_assistant1` 组合）。
- 只允许 3 种协议：`bos0_assistant0`、`bos1_assistant0`、`bos1_assistant1`。
- 每个协议独立文件，不得混写。

**定长索引公式**:
- `record_size = n_layers * hidden_dim * 2`
- `record_offset = token_id * record_size`
- `value_offset = record_offset + ((layer_idx * hidden_dim + dim_idx) * 2)`

**构建与容错规则**:
- 首次构建先预分配主文件总长度（一次性占位，避免运行中扩容碎片）。
- 每个 token 完成写入后更新 `done` 标记；中断后根据 `done` 继续补写。
- 读取时若目标 token 未完成（`done=0`），返回“缺失”而不是报错终止。
- 全部 token 写满后，`done` 文件可保留用于审计，也可删除。
- 一次任务只允许构建一个协议文件；禁止单次任务同时构建 3 个协议。
- 允许分三次构建三个协议文件，按需执行，避免一次性长任务。

**并行效率规则（单协议任务内）**:
- 采用流水线并行：`Reader/Tokenizer` → `GPU Worker` → `Writer`。
- GPU 负责持续前向，写盘异步执行，避免“算完再写、写完再算”的串行空转。
- 建议使用有界队列（防止内存膨胀）和 micro-batch（提升 GPU 利用率）。

**读穿透规则（Probe 统一入口）**:
- Probe/Study 不直接决定“读缓存还是跑模型”，统一调用 `hidden_store` 接口。
- `hidden_store` 优先读取协议缓存：
  - 命中：直接返回隐状态。
  - 未命中：触发一次模型前向计算，写回缓存后返回（read-through）。
- 上层始终“只调用 hidden_store”，分叉逻辑只存在于 `hidden_store` 内部。

**构建入口（CLI + 函数）**:
- CLI 命令统一放在 `src/main.py`：
  - `python -m src.main build-token-hidden-store --bos true --assistant false`
- 可选参数：
  - `--bos`：`true/false`
  - `--assistant`：`true/false`（当 `bos=false` 时必须为 `false`）
  - `--limit`：本次最多构建多少个 token（`0` 表示不限）
  - `--start-token-id`：从指定 token_id 继续构建
- 对应实现函数位于 `src/utils/token_hidden_store.py`：
  - `protocol_from_flags(bos, assistant)`：将参数映射到协议名
  - `build_store_for_protocol(...)`：执行单协议构建，并输出统计信息

---

### 5️⃣ pipeline.py - 处理管道

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

### 9️⃣ src/utils/hooks.py - 模型操作层（第2层）

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
from src.utils.hooks import extract_layer_hidden_states, disable_attention_heads

# 提取隐状态
states = extract_layer_hidden_states(model, layer_indices=[5, 10], input_ids=input_ids)

# 关闭头
disabler = disable_attention_heads(model, head_indices=[0, 1], layer_indices=[5])
# 执行推理...
disabler.remove_all_hooks()
```

**临时参数修改示例**:
```python
from src.utils.hooks import temporary_parameter_patch

with temporary_parameter_patch(model, patches):
    # 在上下文内部运行修改后的模型
    outputs = model(**inputs)

# 退出上下文后，参数必须已经恢复到调用前状态
```

---

