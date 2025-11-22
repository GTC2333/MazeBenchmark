# MazeBenchmark
一个用于评测大模型空间推理与路径规划能力的开源基准，包含：
- 2D 文本迷宫（MazeBench-2D）：以文本网格为输入，路径坐标列表为输出
- 2D 图像迷宫（MazeBench-2D-Image）：以PNG图像为输入，路径坐标列表为输出（多模态）

支持端到端自动化：生成 → 提示构造 → 模型调用 → 输出解析 → 验证与评分 → 报告生成。适配 CI/CD，与外部API解耦（默认使用本地 Mock 适配器）。

## 快速开始

统一运行（推荐）：
```
pip install -r requirements.txt
python bench.py
```

所有输出将统一写入 outputs/：
- text2d_summary.json / text2d_summary.pdf / text2d_report_*.html
- image2d_summary.json / image2d_summary.pdf / image2d_report_*.html / image2d_maze_*.png

文本 2D（独立运行）：
```
pip install -r MazeBench-2D/requirements.txt
python MazeBench-2D/cli.py --model gpt-4o --size 10x10 --workers 4
open MazeBench-2D/examples/report_gpt-4o_10x10.html
```

图像 2D（多模态，独立运行）：
```
pip install -r MazeBench-2D-Image/requirements.txt
python MazeBench-2D-Image/cli.py
open MazeBench-2D-Image/examples/report_10x10_0.html
```

若未配置 OPENAI_API_KEY 等外部密钥，系统将自动使用 MockAdapter，确保流水线在离线/CI 环境可运行。

## 两种模式的差异与一致性
- 一致参数：宽高、障碍密度、陷阱比例、随机种子、最短路径计算与连通性保证
- 差异输入：文本网格 vs PNG 像素化迷宫（标注绿色起点、红色终点）
- 一致评测：解析容忍多格式、路径合法性与最优性验证、S/Q/R/A 四维评分、HTML 报告

## 多模态适配器
- 默认 MockAdapter：在无外部API时生成一个可解析但不一定可行的路径，验证与评分仍然执行
- OpenAIAdapter（可选）：设置 OPENAI_API_KEY 后，支持 vision chat（文本+图像）输入

## 目录结构
- MazeBench-2D：文本模式实现与说明
- MazeBench-2D-Image：图像模式实现与说明（maze_gen、eval_core、model_gateways、report、config、examples）

## 3D 迷宫 Benchmark 规划（Roadmap）
- 阶段 A：3D 网格生成与连通性（体素迷宫，支持 10×10×10 至 40×40×40）
- 阶段 B：最短路径与导航任务定义（含上/下/前/后/左/右六自由度）
- 阶段 C：输入形式
  - 文本：切片序列或层叠投影
  - 图像/渲染：体素渲染为序列帧或俯视/等距视图
- 阶段 D：评测指标扩展
  - 成功率/最优性/鲁棒性/合规性（扩展到三维路径）
  - 可视化报告：体素路径热力图与层切片对比
- 阶段 E：多模态接口
  - 统一适配器协议，支持上传多帧图像
  - 防作弊策略升级：输入扰动与输出沙盒扩展到三维
- 阶段 F：CI 与示例
  - 提供 mock 适配器三维基线，确保无外部依赖也能跑通
  - 与 2D 模式参数保持一致，增加三维特有配置

欢迎贡献与讨论，共同完善多模态空间推理评测。 

# MazeBench-2D

一个用于自动化评测大语言模型在二维文本迷宫任务中的空间推理与规划能力的开源项目。

## 快速开始（3行）
```
pip install -r MazeBench-2D/requirements.txt
python MazeBench-2D/cli.py --model gpt-4o --size 10x10 --workers 4
open MazeBench-2D/examples/report_gpt-4o_10x10.html
```

## 评测流水线
生成 → 输入构造 → 模型调用 → 输出解析 → 多维验证 → 报告生成，全流程无需人工干预，适配 CI/CD。

## 模块结构
- maze_gen：参数化生成标准化迷宫，支持 5×5 至 40×40，含障碍密度与陷阱比例；BFS 连通性验证并输出 shortest_path 与 trap_zones。
- eval_core：多格式解析器（容忍坐标列表/方向序列/含注释/行列混淆/JSON数组/括号变体/纯数字对），四层验证（有效性→陷阱检测→最优性→鲁棒性），动态加权评分 S/Q/R/A。
- model_gateways：统一模型适配（OpenAI/Anthropic），使用 requests，无深度学习框架。
- report：生成交互式 HTML 报告（综合得分、雷达图、热力图、路径对比与失败快照）。
- config：默认参数与防作弊策略（输入扰动 + 输出沙盒）。


## 离线/CI 模式
- 若未设置 OPENAI_API_KEY/ANTHROPIC_API_KEY，CLI 将自动回退到 MockAdapter，确保流水线在 CI 环境无外部服务也可运行。
- 显式使用本地模型：`python MazeBench-2D/cli.py --model mock --size 10x10` 或 `--model mock-gpt-4o`。
- 报告与摘要同样生成于 examples/，用于验证端到端可用性。

## 指标定义
设 S=成功率，Q=最优性，R=鲁棒性，A=防作弊合规；权重随迷宫尺寸动态调整。
- S = 1{路径合法且到达终点}
- Q = 1{长度等于最短路径}；若路径合法但非最短则 0.5
- R = 1{微扰后贪心可纠正}
- A = 1{通过输出沙盒与策略检查}
综合得分 = w_S·S + w_Q·Q + w_R·R + w_A·A，报告中以百分制显示。

## 防作弊机制
- 输入扰动：在提示中加入 nonce 与轻微结构扰动，避免硬编码模式匹配。
- 输出沙盒：剥离非坐标字符，仅保留括号/逗号/数字，限制注入与说明文本。
- 输出限定：仅接受纯路径，拒绝解释性文本。

## 人类基线参考


## 示例输出
在 examples/ 目录将自动生成 GPT-4o 在 10x10 迷宫上的完整评测报告（HTML）。

