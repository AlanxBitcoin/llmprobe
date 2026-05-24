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

# 全词表隐状态定长缓存（统一入口，默认启用）
hidden_store:
  protocol: "bos1_assistant0"        # bos0_assistant0 | bos1_assistant0 | bos1_assistant1
  data_file: "data/cache/hidden_states.{protocol}.f16.bin"
  progress_file: "data/cache/hidden_states.{protocol}.done.bin"
  dtype: "float16"
  n_layers: 36
  hidden_dim: 4096
  token_count_source: "max_token_id_plus_one"
  preallocate: true
  build_one_protocol_per_run: true
  forbid_all_protocols_in_one_run: true
  # 单协议任务内并行（提高 GPU 利用率）
  pipeline_parallel: true
  reader_workers: 2
  writer_workers: 1
  queue_size: 512
  micro_batch_size: 64
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
from src.utils.extract_hidden import (
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

#### 全词表缓存随机读取（定长寻址）
```python
record_size = 36 * 4096 * 2  # float16
record_offset = token_id * record_size
value_offset = record_offset + ((layer_idx * 4096 + dim_idx) * 2)
```

#### 全词表缓存读穿透（统一调用）
```python
# 上层只调这一个接口
hidden = hidden_store.get_hidden_state(
    token_id=token_id,
    protocol=config["hidden_store"]["protocol"],
    layer_idx=target_layer,
)
# hidden_store 内部：
# 1) 命中缓存 -> 直接返回
# 2) 未命中 -> 跑模型 -> 回写缓存 -> 返回
```

#### 模型操作 (utils/hooks.py)
```python
from src.utils.hooks import (
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

