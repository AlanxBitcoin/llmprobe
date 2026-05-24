# Prompt Variants

## 产品需求文档版

```text
项目名称
基于 Llama 3 8B Instruct 的英文单词隐藏层探针与传播分析系统

一、项目背景
希望构建一个本地运行的小型解释性分析系统，用于研究大语言模型对英文单词及多词输入的内部表征。系统需要支持隐藏层分析、属性探针、传播路径可视化、批处理、录屏、结果汇总和总览浏览。

二、项目目标
1. 支持单个英文单词的第 8 层隐藏状态分析
2. 支持多个英文单词的全层传播分析
3. 支持 100 个测试词的批处理实验
4. 支持属性探针训练与属性预测
5. 支持图片、JSON、录屏、索引页、总览页输出
6. 支持后续扩展为更强的 probe 和更细的传播分析

三、用户场景
1. 研究者输入一个词，例如 apple，希望看到：
- 第 8 层高激活维度
- 对应概念类型和概念值候选
- category/color/shape/taste 等属性预测
- 一张综合解释图

2. 研究者输入多个词，例如 apple banana orange，希望看到：
- 全层激活变化
- 层间热力图
- 简化传播线路图
- 保存的图片和数据

3. 研究者对 100 个词进行批量实验，希望得到：
- 全部分析图片
- JSON 结果
- 批处理录屏
- 可筛选的索引页
- 可统计的总览页

四、功能需求

4.1 模型加载
- 使用本地 Hugging Face 兼容目录的 Meta-Llama-3-8B-Instruct
- 支持默认稳定配置
- 支持 4bit 配置
- 本地推理运行

4.2 单词分析
输入：单个裸英文单词
处理：
- 提取第 8 层隐藏状态
- 找出 top-k 高激活维度
- 做概念类型和概念值相似度匹配
- 做属性 probe 预测
输出：
- 综合解释图
- JSON 数据

4.3 多词传播分析
输入：两个及以上裸英文单词
处理：
- 提取全部层隐藏状态
- 选取高激活区域
- 生成跨层热力图
- 生成简化传播线路图
输出：
- 图片
- JSON 数据

4.4 批处理
- 单词模式：每次读取 1 个词，执行单词分析直到文件结尾
- 多词模式：每次读取固定批大小 N 个词，执行多词传播分析直到文件结尾
- 支持全过程录屏导出

4.5 Probe 能力
- 主类别线性 probe
- 属性家族 probe
- 单词属性预测接口
- 输出 top-1 和 top-k 候选及分数

4.6 结果浏览
单词批处理输出：
- summary.csv / summary.json
- index.html
- overview.html / overview.json

多词批处理输出：
- summary.csv / summary.json
- index.html
- overview.html / overview.json

4.7 索引页能力
单词索引页需要支持：
- 缩略图
- 核心属性标签
- 按词筛选
- 按 category/color/shape/taste 筛选
- 按置信度阈值筛选
- 只看高置信度结果

多词索引页需要支持：
- 批次列表
- 图片和数据链接
- 总览跳转

4.8 总览页能力
单词总览页统计：
- 高频 category
- 高频 color
- 高频 shape
- 高频 taste
- 最高置信度预测
- 低置信度复核对象

多词总览页统计：
- 批次数
- batch size 分布
- strongest layer 分布
- strongest batch
- review candidates

五、输入数据
- 100 个常用英文单词测试文件
- 概念类型词表
- 概念值词表
- 概念目录
- 类别标签文件
- 属性标签文件

六、输出结果
- 单次分析图片
- 单次分析 JSON
- 批处理图片
- 批处理 JSON
- CSV 汇总
- HTML 索引页
- HTML 总览页
- MP4 录屏

七、非功能要求
- 本地可运行
- 结构清晰
- 易于扩展
- 支持后续替换模型、词表、层号、probe 配置
- 优先可用性，再逐步增强研究深度

八、迭代优先级
第一阶段
- 单词分析
- 多词分析
- 批处理
- 图片和 JSON
- MP4 导出

第二阶段
- Probe 训练与预测
- 索引页
- 总览页
- 筛选功能

第三阶段
- 更强传播线路图
- 更强属性 probe
- 更细粒度解释
```

## 给代码模型用的精简版

```text
实现一个本地运行的英文单词隐藏层探针系统，基于本地 Hugging Face 兼容目录的 Meta-Llama-3-8B-Instruct。

要求：

1. 单词分析
- 输入一个裸英文单词
- 提取第 8 层 hidden state
- 选出 top-k 高激活维度
- 做概念类型和概念值匹配
- 做属性 probe 预测
- 输出综合图和 JSON
- 核心属性至少包括：
  category, color, shape, taste

2. 多词分析
- 输入两个及以上裸英文单词
- 提取全部层 hidden states
- 生成层-词热力图
- 生成简化传播线路图
- 保存图片和 JSON

3. 批处理
- 从 100 词文本文件读取
- 单词模式：逐词分析
- 多词模式：固定 batch size 分组分析
- 输出全过程录屏，优先 mp4

4. Probe
- 线性 probe
- 主类别分类 probe
- 属性家族 probe
- 单词属性预测命令

5. 输出文件
- 图片
- JSON
- summary.csv
- summary.json
- index.html
- overview.html
- overview.json
- mp4

6. 单词索引页
- 缩略图
- 核心属性标签
- 按 word/category/color/shape/taste 筛选
- 按置信度阈值筛选

7. 单词总览页
- 高频 category/color/shape/taste
- 最高置信度预测
- 低置信度复核对象

8. 多词总览页
- strongest layer 分布
- strongest batch
- review candidates

9. 工程要求
- 本地运行
- 支持默认配置和 4bit 配置
- 先做最小可用版本，再逐步增强
```

## 给研究助手用的实验任务版

```text
实验任务：使用本地 Meta-Llama-3-8B-Instruct 对英文单词做隐藏层探针与传播分析。

一、实验目标
1. 研究单个英文单词在模型第 8 层的内部表征
2. 研究多个英文单词在全部层中的传播模式
3. 评估模型隐藏表征是否可用于预测语义属性

二、实验对象
- 100 个常用英文单词
- 词覆盖 fruit、animal、color、shape、food、vehicle、object、emotion、material、action 等类别

三、实验内容

任务 A：单词层 8 表征分析
对每个输入单词：
- 提取第 8 层 hidden state
- 找出高激活维度
- 用概念词表做相似度匹配
- 用 probe 预测属性
- 保存图和 JSON

重点关注：
- category
- color
- shape
- taste

任务 B：多词传播分析
对每组输入词：
- 提取全部层 hidden states
- 计算每层强度
- 生成热力图
- 生成简化传播线路图
- 保存图和 JSON

任务 C：Probe 实验
1. 训练主类别线性 probe
- 输入：第 8 层向量
- 输出：类别标签
- 记录 accuracy 和预测结果

2. 训练属性家族 probe
- 对 category、color、shape、taste、material、emotion、action_type 分别建模
- 记录每个属性家族的 accuracy 和典型错例

3. 单词属性预测
- 对指定词生成属性预测结果
- 保存每个属性的 top-1 和 top-k 候选

四、批处理流程
1. 单词模式
- 逐词处理 100 个词
- 生成单词综合解释图
- 生成 summary/index/overview
- 导出录屏

2. 多词模式
- 固定 batch size 分组
- 生成传播图
- 生成 summary/index/overview
- 导出录屏

五、结果分析重点
1. 哪些类别最容易被准确预测
2. 哪些属性家族最稳定
3. 哪些词具有高置信度核心属性
4. 哪些词或词组需要人工复核
5. 多词传播中 strongest layer 集中在哪些层

六、交付物
- 单词图
- 多词图
- JSON 数据
- probe 指标
- summary.csv
- index.html
- overview.html
- mp4 录屏

七、研究结论建议
最终应总结：
- 第 8 层是否已包含明显的语义属性信息
- 哪些属性最容易从隐藏层恢复
- 多词传播图是否能反映词间结构性差异
- 当前方法的局限和下一步改进方向
```
