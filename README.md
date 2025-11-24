# Maze Benchmark

This repository provides unified 2D maze benchmarks for LLMs with two modes:
1) Text2D: text-only input/output
2) Image2D: image input + text output (multimodal)

Both modes share a common generation core (structure-first; no legacy density), unified configuration (including start_goal placement), execution flow, and output structures.

## Running

Single entrypoint:

    python bench.py

Configuration at config/config.yaml. Secrets can be placed in a local.yaml ignored by git.

Flags through config keys:
- model: adapter name (mock or provider-specific)
- text2d.size: e.g. '11x11' (recommended odd×odd)
- image2d.size: e.g. '11x11'; image2d.n: number of mazes (recommended odd×odd)
- output_dir: where outputs/ are stored
- text2d.algorithm: dfs or prim
- image2d.algorithm: dfs or prim

- generate_only: if true, only generate and save mazes
- pre_generated_dir: path to a directory with saved mazes to evaluate without generating
- mode: text2d/image2d/all to scope generation or evaluation

Outputs:
- JSON summaries per mode: text2d_summary.json, image2d_summary.json
- HTML reports per run with details
- PDF summaries via ReportLab: text2d_summary.pdf, image2d_summary.pdf
- overview.json aggregating averages

Generate-only saves assets under outputs/mazes_text2d or outputs/mazes_image2d.
Pre-generated evaluation consumes assets named text2d_maze_* or image2d_maze_*.json/.png in the specified directory.

## 3D Roadmap

Planned 3D maze benchmark:
- Voxels-based 3D grid with start/goal on different floors
- Rendering: simple orthographic slices + isometric projection images
- Input: image stacks or short video; Output: textual path or command sequence
- Parameters (structure-first, no density): corridor width, branch frequency, turn bias, loop bias, verticality ratio, entrance/exit placement, dead-end frequency, trap ratio
- Evaluation: shortest-path validation, collision checks, step budget, time-to-solve
- Adapters: extend current interface to pass video frames

Phases:
1) 3D generator and renderer (mock assets)
2) Parser/validator for 3D paths
3) Metrics and reporting unified with 2D
4) Multimodal adapters (image stack/video)
5) End-to-end integration in bench.py

## Parameter effects and realism

New generation method (no density):
- Main path via random walk with local turn bias; avoids long straights and diagonal dominance
- Branching tree style: sample along main path to carve short branches ending in dead-ends
- Wall dispersion: break up large continuous blocks and limit global wall ratio

Key parameters:
- trap_ratio: percentage of trap cells; increases path risk and penalizes unsafe routes
- seed: random seed for reproducibility; use varied seeds for diversity
- cell_px (Image2D): pixel size per cell; larger values create clearer visuals; final image equals maze grid (no outer border)
- start/goal placement: corners vs random; affects path length distribution

Recommended ranges:
- trap_ratio: 0.0–0.2 depending on difficulty
- size: 11x11–41x41 (odd×odd recommended) for practical benchmarking

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
### 使用 Conda 配置运行环境（可选）
如果你倾向于使用 Conda 进行隔离环境管理，可参考以下步骤：

- 创建并激活环境（Python 3.10+ 建议使用 3.11/3.12）
```
conda create -n mazebench python=3.11 -y
conda activate mazebench
```

- 安装依赖（顶层统一运行推荐）
```
pip install -r requirements.txt
```

- 独立运行 Text2D 或 Image2D（如需）
```
# Text2D 仅需 requests
pip install -r MazeBench-2D/requirements.txt

### 配置与命令行示例（算法与种子）

在 config/config.yaml 中选择算法与种子：
```
text2d:
  size: '11x11'
  algorithm: dfs
  seed: 123
image2d:
  algorithm: prim
  seed: 123
  size: '11x11'
  n: 5
```

命令行（Text2D 独立运行）：
```
python MazeBench-2D/cli.py --model mock --size 21x21 --algorithm prim --start_goal random --seed 42
```

命令行（Image2D 独立运行）：
```
python MazeBench-2D-Image/cli.py --size 21x21 --n 3 --algorithm dfs --start_goal corner --seed 100
```

在 bench.py 的 generate_only 模式下，会读取上述配置并生成固定种子可复现的样例。

# Image2D 仅需 requests（核心依赖已在顶层 requirements 安装）
pip install -r MazeBench-2D-Image/requirements.txt
```

- 运行
```
# 统一入口
python bench.py

# 或者分别：
python MazeBench-2D/cli.py --size 11x11 --algorithm dfs
python MazeBench-2D-Image/cli.py --size 11x11 --algorithm prim
```

- 退出环境
```
conda deactivate
```

备注：
- 部分环境在安装 Pillow 或 reportlab 时可能触发编译或下载较慢，建议优先使用官方 Conda Python 版本以及国内镜像源（如通过 pip config 设置 index-url）。
- 若安装受限，可先使用 --model mock 进行离线验证，或仅执行生成（generate_only）。

- text2d_summary.json / text2d_summary.pdf / text2d_report_*.html
- image2d_summary.json / image2d_summary.pdf / image2d_report_*.html / image2d_maze_*.png

文本 2D（独立运行）：
```
pip install -r MazeBench-2D/requirements.txt
python MazeBench-2D/cli.py --model gpt-4o --size 11x11 --workers 4
open MazeBench-2D/examples/report_gpt-4o_11x11.html
```

图像 2D（多模态，独立运行）：
```
pip install -r MazeBench-2D-Image/requirements.txt
python MazeBench-2D-Image/cli.py --size 11x11 --algorithm dfs
open MazeBench-2D-Image/examples/report_11x11_0.html
```

图像 2D（多模态，独立运行）：
```
pip install -r MazeBench-2D-Image/requirements.txt

### 迷宫生成算法（可切换，易扩展）

公共生成核心 common/maze_generator.py 提供算法注册与调度，当前内置：
- dfs：步长为 2 的“挖墙式”回溯（stride-2 DFS），在偶数坐标格点上行走，并打通相邻格之间的墙，形成经典墙体+通道的迷宫（墙=1，路=0）。
- prim：步长为 2 的随机 Prim 生长（stride-2 Prim），维护前沿集合，随机选择前沿并与已开通区域相连，同时打通中间墙格，保证生成树性质。

选择方式：
- 在 config/config.yaml 中设置 text2d.algorithm / image2d.algorithm（dfs 或 prim）
- 或通过 CLI：
  - Text2D: python MazeBench-2D/cli.py --algorithm dfs
  - Image2D: python MazeBench-2D-Image/cli.py --algorithm prim

扩展新算法：
- 在 common/maze_generator.py 中新增 _apply_youralgo 与内部 carve 逻辑
- 通过 CommonMazeGenerator.register_algo('youralgo', handler) 注册到算法表（handler 接口：handler(grid, start, goal)）
- Text2D 与 Image2D 会自动支持该算法（共享生成核心）

可复现性与 AntiCheat：
- 所有生成调用以 config 中 seed 为基准，生成输出中包含 nonce=seed，防作弊策略会读取此 nonce 作为扰动与沙盒的一致随机源
- 相同参数与 seed 下，多次运行将得到完全一致的迷宫与最短路径

参数对齐：
- Text2D 与 Image2D 均接受 width/height/seed/start_goal/algorithm；Image2D 额外多一个 cell_px（像素尺寸）。
- trap_ratio：两种模式均接受；Text2D 会在网格上注入陷阱区域并在验证中使用；Image2D 目前仅为参数对齐，占位未用于渲染与验证。

- 批量数量 n：
  - text2d.n：生成/评估多份文本迷宫（基于 seed 采用 base_seed+i 递增）
  - image2d.n：生成/评估多份图像迷宫（同样采用 base_seed+i 递增）

## 选择迷宫生成算法（DFS/Prim）

两种算法均已内置于公共生成核心 common/maze_generator.py，并通过配置或 CLI 参数选择：

- DFS（默认）：偏置的自避免深度优先挖掘，先主路径，后有限分支，并引入“柱子”抑制大开阔区，生成树状走廊，分支较多、死胡同明显。
- Prim：随机 Prim 风格生长，从起点扩展生成生成树，通道更均匀，环较少（树结构）。

配置文件方式：
```
text2d:
  size: '11x11'
  algorithm: prim
image2d:
  size: '11x11'
  algorithm: dfs
```

命令行方式：
- 文本 2D：`python MazeBench-2D/cli.py --size 21x21 --algorithm prim`
- 图像 2D：`python MazeBench-2D-Image/cli.py --size 21x21 --algorithm dfs`

统一入口 bench.py 也会读取 config/config.yaml 中的 text2d.algorithm / image2d.algorithm，并在 generate_only 模式下写入对应算法生成的迷宫。
统一入口 bench.py 会读取 config/config.yaml 中的参数，包括：
- text2d.algorithm / image2d.algorithm：迷宫生成算法（dfs/prim）
- text2d.seed / image2d.seed：随机种子（用于可复现性）。在批量生成时，系统会使用 base seed + 索引 的策略，例如 image2d.seed=123，生成第 i 个样例使用 seed=123+i。
并且在 generate_only 模式下写入对应算法与种子的迷宫到 outputs/mazes_* 目录。


python MazeBench-2D-Image/cli.py
open MazeBench-2D-Image/examples/report_11x11_0.html
```

若未配置 OPENAI_API_KEY 等外部密钥，系统将自动使用 MockAdapter，确保流水线在离线/CI 环境可运行。

## 两种模式的差异与一致性
- 一致参数：宽高、陷阱比例、随机种子、最短路径计算与连通性保证（不使用密度参数）
- 差异输入：文本网格 vs PNG 像素化迷宫（标注绿色起点、红色终点）
- 一致评测：解析容忍多格式、路径合法性与最优性验证、S/Q/R/A 四维评分、HTML 报告

## 多模态适配器
- 默认 MockAdapter：在无外部API时生成一个可解析但不一定可行的路径，验证与评分仍然执行
- OpenAIAdapter（可选）：设置 OPENAI_API_KEY 后，支持 vision chat（文本+图像）输入

## 代码文件结构与说明

顶层目录：
- bench.py：统一入口，读取配置，调度 Text2D / Image2D 两种模式的生成、评测与报告生成；支持 generate_only 与从预生成资产评测。
- requirements.txt：运行依赖列表（numpy、Pillow、PyYAML、reportlab、requests、tqdm）。
  - config.yaml：主配置文件，包含 text2d.* 与 image2d.* 参数（包括 algorithm 与 seed），以及统一输出目录等。

- config/
  - config.yaml：主配置文件，包含 text2d.* 与 image2d.* 参数（包括 algorithm 与 seed），以及统一输出目录等。
  - local.yaml（可选）：本地私密配置，gitignore 已忽略；可覆盖部分密钥或参数。
- common/
  - maze_generator.py：公共迷宫生成核心，提供可扩展的算法注册与调度，当前支持 'dfs' 与 'prim'；统一生成网格、起终点与 shortest_path。
  - config_loader.py：加载与合并配置（config.yaml + local.yaml + 环境变量），并为 bench.py 提供字典形式的最终配置。
- outputs/：统一输出目录；包含 mazes_text2d/ 与 mazes_image2d/ 子目录用于 generate_only 的资产保存。

文本模式（MazeBench-2D/）：
- cli.py：文本模式的命令行入口，支持 --size, --model, --algorithm 等参数；运行后在 examples/ 生成报告。
- maze_gen/generator.py：文本迷宫生成器，封装 MazeConfig（含 algorithm）并调用 common 生成核心，输出网格与 shortest_path。
- eval_core/：文本输出解析与验证模块，包含路径解析器、合法性与最优性验证、评分逻辑；与报告模块协作生成评测摘要。
- report/：HTML 报告生成模块，绘制热力、雷达与对比可视化。
- model_gateways/：模型适配接口（Mock/OpenAI/Anthropic 等），通过统一协议调用模型并返回路径文本。
- config/：模式内的默认参数与示例配置。
- examples/：运行示例输出目录（HTML 报告与 JSON 摘要）。

图像模式（MazeBench-2D-Image/）：
- cli.py：图像模式命令行入口，支持 --size, --algorithm 等；运行后在 examples/ 生成报告与PNG。
- maze_gen/generator.py：图像迷宫生成器，封装 MazeConfig（含 algorithm）并调用 common 生成核心，随后渲染为 PIL 图片（标注起点与终点）。
- eval_core/：图像模式下的输出解析与验证（与文本模式保持一致的接口）。
- report/：HTML 报告生成模块（图像版）。
- model_gateways/：多模态模型适配接口（Mock/OpenAI 等）。
- config/：模式内默认参数与示例配置。
- examples/：运行示例输出目录（HTML/JSON/PNG）。

扩展点（算法）：
- 在 common/maze_generator.py 中新增 _apply_xxx 与对应算法实现，并在 algo_map 注册：
  - 'dfs'：递归回溯/深度优先风格，先主路径后分支，辅以柱体控制开阔度。
  - 'prim'：随机 Prim 生长，保持生成树结构与较均匀的通道分布。
- 新算法只需实现 carve 逻辑并注册到 algo_map，即可被两种模式与 bench.py 统一调用。

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
python MazeBench-2D/cli.py --model gpt-4o --size 11x11 --workers 4
open MazeBench-2D/examples/report_gpt-4o_11x11.html
```

## 评测流水线
生成 → 输入构造 → 模型调用 → 输出解析 → 多维验证 → 报告生成，全流程无需人工干预，适配 CI/CD。

## 模块结构
- maze_gen：参数化生成标准化迷宫，支持 5×5 至 40×40；不再使用密度参数，转为结构优先（自避免主路径、受限分支、柱体保留），保证连通并输出 shortest_path 与 trap_zones。
- eval_core：多格式解析器（容忍坐标列表/方向序列/含注释/行列混淆/JSON数组/括号变体/纯数字对），四层验证（有效性→陷阱检测→最优性→鲁棒性），动态加权评分 S/Q/R/A。
- model_gateways：统一模型适配（OpenAI/Anthropic），使用 requests，无深度学习框架。
- report：生成交互式 HTML 报告（综合得分、雷达图、热力图、路径对比与失败快照）。
- config：默认参数与防作弊策略（输入扰动 + 输出沙盒）。


## 离线/CI 模式
- 若未设置 OPENAI_API_KEY/ANTHROPIC_API_KEY，CLI 将自动回退到 MockAdapter，确保流水线在 CI 环境无外部服务也可运行。
- 显式使用本地模型：`python MazeBench-2D/cli.py --model mock --size 11x11` 或 `--model mock-gpt-4o`。
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
在 examples/ 目录将自动生成 GPT-4o 在 11x11 迷宫上的完整评测报告（HTML）。

