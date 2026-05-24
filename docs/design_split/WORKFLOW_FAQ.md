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
   - HOOKS_USAGE.md (如果涉及 src/utils/hooks.py)
   - API 文档
   - 示例代码
   ```

### 修改示例

#### 示例 1: 添加新的隐状态提取模式

**步骤 1**: 修改设计文档
```
在"隐状态提取三种模式对比"中添加新模式
更新 src/utils/extract_hidden.py 的职责描述
```

**步骤 2**: 修改代码
```python
# src/utils/extract_hidden.py
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
在"src/utils/hooks.py"模块说明中添加新函数
更新数据流图
```

**步骤 2**: 修改代码
```python
# src/utils/hooks.py
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

## 版本日志

- 历史版本与每版变更已迁移到独立文档：
  - `docs/PROJECT_DESIGN_CHANGELOG.md`
- 维护规则：
  - 每次改 `PROJECT_DESIGN.md` 时，必须同时在 changelog 新增一条版本记录（倒序追加）。
  - changelog 不覆盖旧条目，只追加新版本，确保可以完整回溯每一版改动。

---

## 常见问题

**Q: 我想添加一个新功能，应该从哪里开始？**
A: 先修改此设计文档，在适当位置添加功能描述，然后根据修改后的设计文档修改代码。

**Q: `src/utils/hooks.py` 和 `src/utils/extract_hidden.py` 有什么区别？**
A: `src/utils/extract_hidden.py` 是高级 API，专为项目的隐状态提取业务设计。`src/utils/hooks.py` 是低级模型操作 API，默认使用第1层已经常驻在 GPU 上的模型，可供任何研究脚本使用。

**Q: 全词表隐状态缓存任务中断了怎么办？**
A: 不需要重头跑。构建器通过 `data/cache/hidden_states.{protocol}.done.bin` 记录每个 `token_id` 的完成状态；恢复时只补写未完成 token。

**Q: BOS 和 assistant 会影响缓存结果吗？**
A: 会。`BOS` 与 `assistant` 前缀都会改变上下文分布，隐状态会显著变化。必须按协议分文件存储，不能混用。

**Q: 为什么不一次性把 3 个协议都跑完？**
A: 设计上明确禁止“单任务三协议同跑”，每次只跑一个协议，便于中断续跑和资源管理。三个协议通过三次任务分开构建。

**Q: 如何提高缓存构建时 GPU 利用率？**
A: 在单协议任务内采用 Reader/GPU/Writer 流水线并行，GPU 连续计算，写盘异步执行，避免串行等待导致空转。

**Q: Probe 是直接跑模型，还是优先查缓存？**
A: Probe 统一只调用 `hidden_store`。`hidden_store` 先查缓存，未命中才临时跑模型并回写缓存，然后返回结果。

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

**Q: 我想修改 `src/utils/hooks.py`，需要更新这个设计文档吗？**
A: 是的。如果是重要改变，应该更新本文档的 "`src/utils/hooks.py`" 部分，然后同步更新 HOOKS_USAGE.md。

---

## 联系方式

有关设计文档的问题，请参考：
- 本文件: `PROJECT_DESIGN.md`
- 模块文档: 各模块的 docstring
- 使用指南: `HOOKS_USAGE.md`, `README.md`
