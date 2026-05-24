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
├─→ src/utils/hooks.py (通过 API 操作常驻模型)
├─→ src/utils/extract_hidden.py (提取隐状态)
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
      ├─→ utils/extract_hidden.py (提取隐状态)
      │   └─→ model forward pass
      │
      ├─→ probes/ (Probe 层，第3层)
      │   ├─→ linear_probe.py
      │   ├─→ attribute_probe.py
      │   └─→ concept_match.py
      │
      └─→ utils/video.py (可视化)

study/ (Study 研究层，第4层)
  ├─→ study_linear_probe.py
  ├─→ study_attribute_probe.py
  │
  ├─→ probes/ (调用 Probe 类)
  │   └─→ 生成 CSV 格式输出
  │
  └─→ utils/hooks.py (通用模型操作 API)
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
from src.utils.hooks import extract_layer_hidden_states

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

### 3️⃣ 扩展 src/utils/hooks.py

**步骤**:
1. 编辑 `src/utils/hooks.py`
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

