# PROJECT_DESIGN Changelog

本文件用于记录 `docs/PROJECT_DESIGN.md` 的独立版本日志。

维护规则：
- 仅追加，不覆盖历史条目。
- 新版本放在最上方（倒序）。
- 每条记录至少包含：版本号、日期、变更摘要。

---

## v1.20 (2026-05-24)
- 文档同步 `build-token-hidden-store` 构建入口：
  - 在 `src/main.py` 增加单协议构建命令，支持 `--bos` 与 `--assistant` 参数。
  - 增加 `--limit` / `--start-token-id` 以支持分段构建和断点续跑。
- 明确代码对应关系：
  - `src/utils/token_hidden_store.py` 提供 `protocol_from_flags(...)` 与 `build_store_for_protocol(...)` 作为构建函数入口。

## v1.19 (2026-05-24)
- 增加 `hidden_store` 读穿透（read-through cache）规则：
  - Probe/Study 统一只调用 `hidden_store`，不直接分支到模型。
  - `hidden_store` 命中缓存直接返回；未命中则临时跑模型并回写后返回。
- 更新数据流与 API 示例，明确“数据库优先、未命中回填”的统一路径。

## v1.18 (2026-05-24)
- 完善全词表隐状态缓存协议：
  - 引入 `bos` / `assistant` 协议约束，仅保留 3 种有效组合。
  - 按协议分主文件与进度文件，禁止跨协议混写。
- 明确构建策略：
  - 一次任务只运行一个协议文件，三个协议分三次构建。
  - 禁止单任务“一次跑完三协议”。
- 增加单协议任务内部并行规范：
  - Reader/Tokenizer → GPU Worker → Writer 流水线并行。
  - 增加 `hidden_store` 并行配置项（队列、worker、micro-batch）。

## v1.17 (2026-05-24)
- 在 `PROJECT_DESIGN.md` 增加“全词表隐状态定长缓存”设计：
  - 单文件定长存储（`hidden_states.f16.bin`）+ 进度标记（`hidden_states.done.bin`）
  - 基于 `token_id` 的偏移公式与随机读取规则
  - 预分配、断点续写、中断恢复与缺失容错约束
- 增加对应配置项 `hidden_store` 与缓存数据流说明。

## v1.16 (2026-05-24)
- 按规范精简 `PROJECT_DESIGN.md`：移除顶部“本次更新”整段内容。
- 约定后续所有版本变更只记录在本 changelog 中，主设计文档保持结构与规则清晰。

## v1.15 (2026-05-24)
- 将版本历史从 `PROJECT_DESIGN.md` 外置到本独立日志文件。
- 在 `PROJECT_DESIGN.md` 中新增“版本日志”入口与维护规则，避免历史版本信息被覆盖。

## v1.14 (2026-05-24)
- 同步 `src/utils/` 迁移后的文档路径：目录树、模块说明、依赖图与代码示例统一为新路径。

## v1.13 (2026-05-24)
- 增加 `src/utils/` 结构，定义工具类模块下沉迁移清单（含 hooks/extract_hidden/utils/video/visualize_*）。

## v1.12 (2026-05-24)
- 增加启动与路径约束：统一 `src` 导入与根目录路径解析，兼容 `python -m src.main` 与 `python src/main.py`。

## v1.11 (2026-05-24)
- 增加版本管理约束：`data/` 全目录忽略，不纳入 Git 版本管理。

## v1.10 (2026-05-24)
- 明确目录规范：`src/main.py` 与 `src/hooks.py` 为唯一位置；`probe/` 根目录不保留 `.py` 文件。

## v1.9 (2026-05-24)
- 收尾 `outputs/` → `data/outputs/` 迁移：明确剩余待改脚本与完成判定标准（先改文档再改代码）。

## v1.7 (2026-05-24)
- 将原第4层 `src/tests/` 研究脚本改名为 `src/study/`。
- 新增 `src/test/` 作为真正的软件测试目录。

## v1.6 (2026-05-24)
- 将默认启动配置统一为 `configs/custom.yaml`。
- 正常启动不需要输入配置文件名，仍支持显式指定其他 YAML。

## v1.5 (2026-05-24)
- 明确应用启动时同时启动 Llama API 和 Web UI server。
- 网页按钮点击后再调用对应 study/probe 底层代码。

## v1.4 (2026-05-24)
- 增加第5层 UI 层：`src/ui/` 本地网页 server、Study/Probe 按钮、参数表单、CSV 结果展示和可追加图表。

## v1.3 (2026-05-24)
- 明确 hooks 临时修改模型参数时必须备份并在调用结束后回退。
- 无法恢复时标记 dirty 并按需重新加载。

## v1.2 (2026-05-24)
- 将第1层和第2层调整为模型运行时 API 模式。
- 模型加载到 GPU 后默认常驻，多次 study 优先复用，必要时才重新加载。

## v1.1 (2026-05-24)
- 将项目主数据输出模式改为 CSV。
- 明确第3层 probes/ 负责每个 Probe 的 CSV 表格格式。

## v1.0 (2026-05-24)
- 初始版本，整合现有项目文档 + hooks.py 模块。
