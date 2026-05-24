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
[3] 预处理词 (utils/extract_hidden.py)
    ├─ Tokenize 词
    ├─ 转移到 GPU
    └─ 返回: tensor inputs
    ↓
[4] 通过 hidden_store 获取隐状态（统一入口）
    ├─ 优先查缓存（按 protocol + token_id）
    ├─ 命中则直接返回 hidden vectors
    ├─ 未命中则执行 model forward pass
    └─ 回写缓存后返回 hidden vectors
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

### 全词表缓存数据流（可选长期任务）

```
启动缓存构建任务
    ↓
[1] 选择一个协议（bos0_assistant0 / bos1_assistant0 / bos1_assistant1）
    ↓
[2] 从 tokenizer 获取 token_id 范围（0..max_token_id）
    ↓
[3] 预分配该协议对应的 hidden_states.*.f16.bin 总长度
    ↓
[4] 单协议流水线并行写入（Reader -> GPU Worker -> Writer）
    ├─ offset = token_id * record_size
    └─ 写入完成后置 done[token_id] = 1
    ↓
[5] 中断恢复时仅扫描 done=0 的 token 继续写
    ↓
[6] 构建完成后，主文件可直接作为离线“只读库”复用
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
    ├─ 如需钩子则调用第2层 utils/hooks.py
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

