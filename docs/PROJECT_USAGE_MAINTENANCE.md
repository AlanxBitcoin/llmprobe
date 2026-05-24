# LLM 探针项目使用及维护说明

本文档用于说明本项目的日常使用、实验流程、输出结果解读和维护规范。项目当前目标是：对本地 `Meta-Llama-3-8B-Instruct` 的隐藏状态进行探针分析，观察单词、词库、短句、词序变化等输入在模型不同层、不同维度上的表示差异。

## 1. 项目定位

本项目不是普通问答程序，而是一个“模型内部表示观察工具”。它主要做四件事：

- 把单词、词库或短句输入本地大模型。
- 提取模型隐藏层信息，例如第 8 层，或全部 32 层 × 4096 维。
- 对隐藏状态做统计、差值、排序、属性近似解释和可视化。
- 保存 CSV、JSON、PNG、HTML、DOCX、MP4 等实验结果。

需要特别注意：模型生成文本和隐藏层探针结果不是一回事。隐藏层探针结果是项目真正分析的对象；生成文本只是用于观察模型接续输出，可配置为关闭。

## 2. 当前环境

推荐环境：

- Windows + PowerShell
- Python / Anaconda
- CUDA 可用
- NVIDIA RTX 4060 或更高显存配置
- 本地 Hugging Face 兼容模型目录

本项目常用 Python：

```powershell
C:\Users\qiang\anaconda3\python.exe
```

本地模型路径曾使用：

```text
D:\杂项\models\Meta-Llama-3-8B-Instruct
```

配置文件中如出现乱码路径，需要手动确认它实际指向正确模型目录。

## 3. 目录结构

项目根目录：

```text
D:\vbcode2026\LLM探针
```

主要目录：

```text
configs/    配置文件
data/       输入词库、概念词表、标签表
docs/       使用说明、需求文档、提示词文档
outputs/    所有实验输出
src/        主项目代码
tools/      专项分析脚本
```

常见根目录脚本：

```text
main.py                         主入口，负责常规单词/词库/属性分析
generate_report.py              报告生成脚本
generate_multi_report.py        多实验报告脚本
generate_shuffle_report.py      顺序打乱实验报告脚本
run_experiments.py              批量实验脚本
run_shuffled_experiments.py     打乱顺序实验脚本
```

专项工具脚本：

```text
tools/analyze_command_word_layers.py
tools/analyze_sequence_word_layers.py
tools/render_command_diff_charts.py
tools/render_full_4096_charts.py
```

## 4. 配置文件

主要配置文件：

```text
configs/default.yaml
configs/quantized-4bit.yaml
```

日常建议优先使用：

```text
configs/quantized-4bit.yaml
```

因为本地运行 `Meta-Llama-3-8B-Instruct` 时，4bit 更省显存。

重点字段：

```yaml
model:
  model_name_or_path: "D:/杂项/models/Meta-Llama-3-8B-Instruct"
  device_map: "auto"
  torch_dtype: "float16"
  load_in_4bit: true
  load_in_8bit: false
  trust_remote_code: true

analysis:
  target_layer: 8
  top_k_dims: 24
```

维护建议：

- 如果模型路径变化，只改配置文件，不要在代码里硬编码。
- 如果 4bit 加载失败，可以临时改为 8bit 或全精度。
- 每次修改配置后，先跑一个小实验确认模型能加载。

## 5. 依赖安装

基础安装：

```powershell
pip install -r requirements.txt
```

当前主要依赖：

```text
torch
transformers
accelerate
bitsandbytes
scikit-learn
sentencepiece
pyyaml
numpy
matplotlib
seaborn
networkx
imageio
imageio-ffmpeg
```

如果视频生成失败，优先检查：

```powershell
pip install --user imageio-ffmpeg
```

## 6. 常规使用流程

建议从小到大运行。

### 6.1 单词第 8 层分析

命令：

```powershell
python main.py --config configs/quantized-4bit.yaml run-single-word apple
```

输出目录：

```text
outputs/single_word/
```

主要输出：

```text
figures/apple.png
data/apple.json
```

适用场景：

- 看单个词在第 8 层哪些维度数值高。
- 看近似属性，如颜色、形状、类别、味道等。
- 对 `asphalt`、`cloud`、`white`、`black` 等词做单独分析。

### 6.2 两词或多词相加

命令：

```powershell
python main.py --config configs/quantized-4bit.yaml run-word-sum asphalt cloud
```

输出目录：

```text
outputs/word_sum/
```

用途：

- 观察两个词隐藏表示相加后的第 8 层激活。
- 对比单词单独输入和组合表示之间的差异。

### 6.3 两词相减

命令：

```powershell
python main.py --config configs/quantized-4bit.yaml run-word-diff asphalt cloud
```

输出目录：

```text
outputs/word_diff/
```

注意：这里是分别输入 `asphalt` 和 `cloud` 后，对隐藏表示做相减，不是把字符串 `asphalt-cloud` 直接输入模型。

### 6.4 多词传播分析

命令：

```powershell
python main.py --config configs/quantized-4bit.yaml run-multi-word apple banana orange
```

输出目录：

```text
outputs/multi_word/
```

用途：

- 观察多个词在多层中的激活传播趋势。
- 绘制简化传输线路图。

### 6.5 100 词批处理

默认词库：

```text
data/word_list_100.txt
```

命令：

```powershell
python main.py --config configs/quantized-4bit.yaml run-single-batch
```

输出目录：

```text
outputs/single_batch/
```

常见输出：

```text
index.html
overview.html
summary.csv
single_batch.mp4
```

### 6.6 多词批处理

命令：

```powershell
python main.py --config configs/quantized-4bit.yaml run-multi-batch --batch-size 3
```

含义：每次从词库读取 3 个词作为一组进行多词传播分析。

输出目录：

```text
outputs/multi_batch_3/
```

### 6.7 全局词库分析

命令：

```powershell
python main.py --config configs/quantized-4bit.yaml run-global-analysis-all
```

默认会扫描：

```text
data/*.txt
```

用途：

- 对英文标点、中文标点、阿拉伯数字、罗马数字等词库做批量分析。
- 对颜色、形状、声音、味觉等词库做整体统计。

## 7. 颜色词库实验

颜色词库文件：

```text
data/color_words.txt
```

命令：

```powershell
python main.py --config configs/quantized-4bit.yaml run-color-words-experiment --word-file data/color_words.txt --run-name color_words
```

当前颜色实验支持：

- 单个颜色词分别输入。
- 全部颜色词一次性输入。
- 第 8 层隐藏状态分析。
- 正负值分开绘图。
- 全 4096 维统计。
- 绝对值均值、最大绝对值、出现次数等统计。
- 图表和视频输出。

常见输出目录：

```text
outputs/color_words/
```

## 8. command word 双文本分析

输入目录：

```text
outputs/command word/
```

典型输入文件：

```text
command word.txt
no command word.txt
```

命令：

```powershell
python tools/analyze_command_word_layers.py --config configs/quantized-4bit.yaml --left "outputs/command word/command word.txt" --right "outputs/command word/no command word.txt" --out-dir "outputs/command word/layer32_4096_diff" --top-k 10 --max-new-tokens 96
```

处理方式：

- 分别输入两个文本。
- 提取 32 层隐藏状态。
- 每层取最后一个 token 的 4096 维向量。
- 两个输入对应层、对应维度相减。
- 对差值取绝对值。
- 每层选 Top10。
- 记录输入 token 和生成 token。
- 输出图表和 HTML。

适用场景：

- 比较有命令和无命令文本的差异。
- 比较两个任务提示在模型内部的不同。

## 9. sequence word 顺序差异分析

输入目录：

```text
outputs/sequence word/
```

当前输入文件：

```text
sequence1.txt
sequence2.txt
```

当前例子：

```text
sequence1.txt: 鞋买大了，裤子买小了
sequence2.txt: 裤子买小了，鞋买大了
```

命令：

```powershell
python tools/analyze_sequence_word_layers.py --config configs/quantized-4bit.yaml --left "outputs/sequence word/sequence1.txt" --right "outputs/sequence word/sequence2.txt" --out-dir "outputs/sequence word/layer32_4096_sequence_mean_diff" --top-k 10 --max-new-tokens 96
```

这个脚本不同于 command word 的处理方式。

处理方式：

- 分别读取两个序列文本。
- tokenizer 分词。
- 加载模型并提取隐藏状态。
- 跳过 embedding，只保留 Transformer 第 1-32 层。
- 排除 BOS token。
- 对每层所有内容 token 的 hidden states 求均值。
- 每个输入得到 `[32, 4096]`。
- 计算 `sequence1 - sequence2`。
- 对差值取绝对值。
- 每层选差异最大的 Top10 维度。
- 输出 CSV、JSON、PNG、HTML。

为什么这样处理：

- 词序变化实验关注整句话的整体表征。
- 如果只取最后一个 token，结果会偏向句尾 token。
- 对全部内容 token 求均值，更适合比较“同一批词，不同顺序”的影响。

当前输出目录：

```text
outputs/sequence word/layer32_4096_sequence_mean_diff/
```

主要输出：

```text
index.html
sequence1_full_32_layers_4096.csv
sequence2_full_32_layers_4096.csv
layer_dim_abs_diff_top10.csv
layer_dim_abs_diff_top10.json
sequence1_prompt_tokens.csv
sequence1_generated_tokens.csv
sequence2_prompt_tokens.csv
sequence2_generated_tokens.csv
generated_tokens.json
layer_top1_abs_diff.png
layer_top10_abs_diff_heatmap.png
layer_top10_abs_diff_lines.png
top_diff_dim_frequency.png
metadata.json
```

## 10. token 记录说明

项目会记录两类 token：

```text
prompt tokens      输入文本被 tokenizer 处理后的 token
generated tokens   模型在输入后继续生成的 token
```

非常重要：

- prompt tokens 是模型真正看到的输入。
- generated tokens 是模型自由续写，不参与隐藏层差值计算。
- 如果输入裸中文短句，模型可能自动翻译、解释或补全，这是模型行为，不是程序额外加了翻译指令。

例如 `sequence1.txt`：

```text
鞋买大了，裤子买小了
```

prompt token ids：

```text
[128000, 122584, 106191, 27384, 35287, 3922, 70892, 97, 45829, 106191, 31809, 35287]
```

其中：

```text
128000 = <|begin_of_text|>
70892 + 97 = “裤”的 byte-level 分片
```

CSV 中某些单 token 显示为 `�` 是正常现象，因为一个汉字可能被拆成多个 byte-level token。需要多个 token 合起来 decode 才能恢复完整汉字。

维护建议：

- 主实验默认重点看 prompt token 和 hidden states。
- generated token 建议作为可选观察项。
- 后续可以增加“合并 byte-level token 显示”的功能，提升中文 token 可读性。

## 11. 隐藏状态统计说明

对单个输入，常见隐藏状态形状：

```text
[batch, seq_len, hidden_dim]
```

对 Llama 3 8B：

```text
hidden_dim = 4096
Transformer 层数 = 32
```

项目通常跳过：

```text
hidden_states[0]
```

因为它是 embedding 输出。

保留：

```text
hidden_states[1:33]
```

即 Transformer 第 1-32 层。

不同实验的聚合方式：

```text
单词第8层分析：通常看目标层的单词表示。
command word：每层取最后一个 token。
sequence word：每层对全部内容 token 求均值。
颜色词库三模式：按单词输入、全部输入等模式分别统计。
```

维护重点：不同聚合方式不能随意混合比较。报告中必须注明本次实验是“最后 token”还是“全部 token 均值”。

## 12. 输出文件怎么看

### 12.1 full_32_layers_4096.csv

这类文件每个输入组通常有：

```text
32 × 4096 = 131072 行
```

常见字段：

```text
layer
dim
sample_count
mean_value
mean_abs_value
max_value
min_value
std_value
positive_count
negative_count
zero_count
mean_direction
```

含义：

- `layer`：Transformer 层号，1-32。
- `dim`：隐藏维度编号，0-4095。
- `mean_value`：该维度的平均值，保留正负。
- `mean_abs_value`：绝对值平均。
- `max_value` / `min_value`：最大值与最小值。
- `positive_count` / `negative_count`：正负出现次数。

### 12.2 layer_dim_abs_diff_top10.csv

这是双输入比较的关键表。

常见字段：

```text
layer
rank
dim
abs_diff
signed_diff
sequence1_mean / command_mean
sequence2_mean / no_command_mean
larger_group
```

含义：

- `abs_diff`：两个输入在同层同维度上的差值绝对值。
- `signed_diff`：保留方向的差值。
- `rank`：该层内部差异排名。
- `dim`：差异维度编号。

注意：

- `abs_diff` 大说明两个输入在该层该维度差异大。
- 不能直接说某个维度等于某个人类概念。
- 维度和概念之间通常是近似、分布式、多维共同表达。

### 12.3 图表文件

常见图表：

```text
layer_top1_abs_diff.png
layer_top10_abs_diff_heatmap.png
layer_top10_abs_diff_lines.png
top_diff_dim_frequency.png
```

怎么看：

- `layer_top1_abs_diff.png`：每层最大差异值，适合看哪一层差异最强。
- `layer_top10_abs_diff_heatmap.png`：每层 Top10 差异维度热力图，适合查具体维度。
- `layer_top10_abs_diff_lines.png`：Top10 差异随层变化，适合看趋势。
- `top_diff_dim_frequency.png`：哪些维度反复进入 Top10，适合找稳定高差异维度。

## 13. 属性表与概念解释

项目中有两类概念资源：

```text
data/concept_types.txt
data/concept_values.txt
data/concept_catalog.yaml
```

以及符号类概念资源：

```text
data/symbolic_concept_types.txt
data/symbolic_concept_values.txt
data/symbolic_concept_catalog.yaml
data/symbolic_token_attributes.yaml
```

维护原则：

- 普通英文单词不应只用符号属性表解释。
- 标点、数字、罗马数字应使用符号属性表或独立解释体系。
- 不要把 `taste:sweet` 这类属性强行套到英文标点上。
- 如果词库类型变化，概念表也要同步扩充或切换。

关于维度解释：

- 单个维度不一定等于 `fruit`、`color`、`shape` 这样的概念。
- 更合理的说法是：某些维度可能与某类概念方向相关。
- 人类概念通常由多个维度共同表达。
- 维度解释需要结合词库、对照实验、相似词、反义词、消融实验一起看。

## 14. 视频输出说明

部分批处理流程支持视频：

```text
single_batch.mp4
multi_batch_3.mp4
```

视频生成方式：

- 不是桌面录屏。
- 是把每一帧分析图按顺序合成为 MP4。

优点：

- 稳定。
- 可复现。
- 不依赖屏幕窗口。

如果视频没生成，检查：

- 配置 `video.enabled` 是否为 `true`。
- 是否安装 `imageio-ffmpeg`。
- 输出目录中是否已有 frame 图像。

## 15. 常见问题

### 15.1 为什么中文 token 有乱码或 `�`

因为 tokenizer 可能把一个汉字拆成多个 byte-level token。单独 decode 某个 token 可能不是完整字符。

处理建议：

- 看 `generated_tokens.json` 中整体 decode 文本。
- 不要只根据单个 token 的 `decoded` 判断字符。
- 后续维护可增加 token 合并显示。

### 15.2 为什么模型会生成英文翻译

因为裸中文句子输入 Instruct 模型后，模型可能自由续写成翻译、解释或改写。程序没有额外加翻译指令。

如需避免：

- 主实验只记录 prompt token，不调用 generate。
- 或把 generate 设为可选。
- 不建议为了约束生成而改 prompt，因为这会改变隐藏层分析输入。

### 15.3 为什么两个实验差异值不一样

先检查聚合方式是否相同：

```text
最后 token
全部 token 均值
第 8 层
32 层全维
单词输入
整句输入
全部词一次输入
```

不同处理方式的结果不能直接做强对比。

### 15.4 为什么某个维度反复出现

反复进入 Top10 说明该维度对当前对比任务敏感，但不等于它就是某个明确概念。

建议继续做：

- 相似词测试。
- 反义词测试。
- 词序变化测试。
- 维度消融。
- 与 logits 或下游输出关联分析。

## 16. 维护规范

### 16.1 新增实验脚本

建议放在：

```text
tools/
```

命名规则：

```text
analyze_<experiment_name>_layers.py
render_<experiment_name>_charts.py
```

脚本需要包含：

- 输入文件参数。
- 输出目录参数。
- 配置文件参数。
- top-k 参数。
- metadata.json。
- index.html。
- CSV / JSON 输出。

### 16.2 修改主流程

主流程代码在：

```text
src/pipeline.py
main.py
```

维护建议：

- 通用能力放到 `src/`。
- 一次性专项实验放到 `tools/`。
- 命令入口稳定后再合并到 `main.py`。

### 16.3 输出命名

推荐命名：

```text
input_a_*
input_b_*
sequence1_*
sequence2_*
color_words_*
```

避免继续使用不准确的：

```text
command_*
no_command_*
```

除非实验确实是 command/no-command。

### 16.4 文档同步

每次新增实验后，至少更新：

```text
docs/PROJECT_USAGE_MAINTENANCE.md
```

如果新增主入口命令，也更新：

```text
README.md
docs/USAGE.md
```

### 16.5 结果可复现

每个输出目录建议保留：

```text
metadata.json
generated_tokens.json
layer_dim_abs_diff_top10.csv
index.html
```

metadata 至少包含：

```text
输入文件
输入文本
模型路径
配置文件
聚合方式
是否包含 BOS
层数
隐藏维度
top_k
```

## 17. 推荐后续改进

优先级较高：

- 把 generate 改成可选，默认只记录 prompt token。
- 增加中文 byte-level token 合并显示。
- 统一双输入比较脚本命名，从 command/no-command 改为 input_a/input_b。
- 为每个实验输出更完整的 metadata。
- 把 sequence word 的处理方式写入 HTML，避免误解。

优先级中等：

- 增加 dim 解释报告模板。
- 增加跨实验维度频次汇总。
- 增加 Word 报告自动生成入口。
- 增加更清晰的 4096 维全景图。

优先级较低：

- 增加交互式网页筛选。
- 增加浏览器自动截图。
- 增加更多模型对比。

## 18. 推荐日常操作顺序

新实验建议按这个顺序：

1. 准备输入词库或文本。
2. 确认配置文件模型路径正确。
3. 先跑小样本。
4. 检查 prompt token 是否符合预期。
5. 检查 hidden states 输出行数是否为 `32 × 4096`。
6. 检查图表是否生成。
7. 再跑完整实验。
8. 写入 metadata 和报告。
9. 归档输出目录。

## 19. 当前重点工作计划

短期：

- 完成 sequence word 顺序差异实验的深度解释。
- 修改生成 token 为可选项，避免把模型续写误解为实验处理。
- 改善中文 token 展示。

中期：

- 建立统一的双输入比较框架。
- 将 command word、sequence word、颜色词库等实验纳入统一报告模板。
- 完成高差异维度跨实验统计。

长期：

- 做更严格的维度-概念验证。
- 增加消融实验和 causal tracing 风格分析。
- 从“可视化观察”逐步推进到“可解释性假设验证”。

