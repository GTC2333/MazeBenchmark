
---

# MazeBench

> 一个可复现、多模态的开源基准测试，用于系统评估大语言模型（LLM）在**二维迷宫导航任务**中的空间推理与路径规划能力。  
> 支持**纯文本网格**与**图像输入**两种模式，端到端自动化，开箱即用。

*专为科研评测、CI/CD 自动化与公平模型对比而设计。*

---

## 🚀 为什么选择 MazeBench？

现有空间推理评测常存在以下问题：
- ❌ 使用固定迷宫（缺乏可控生成）  
- ❌ 基于“墙密度”生成（导致不自然的大片空地）  
- ❌ 文本与图像模式输入/输出不统一  
- ❌ 忽略鲁棒性与防作弊设计  

**MazeBench 的优势**：
- ✅ **结构优先生成**：主路径 + 受控分支 + 陷阱区，更贴近真实环境  
- ✅ **统一核心**：文本与图像模式共享生成、验证、评分逻辑（S/Q/O/A 四维指标；R 暂停）  
- ✅ **端到端流水线**：生成 → 构造提示 → 模型调用 → 解析 → 验证 → 报告，全自动  
- ✅ **防作弊 & 可复现**：基于 `nonce` 的输入扰动 + 种子锁定  
- ✅ **CI/CD 友好**：默认使用 Mock 适配器，无需外部 API 即可运行  

---

## 模式

| 模式 | 输入 | 输出 | 适用场景 |
|------|------|------|----------|
| **Text2D** (`MazeBench-2D`) | 文本网格（`0`=通路，`1`=墙，`S`/`G`=起终点） | 文本路径（如 `[(1,1), (1,2), ...]`） | 纯推理能力评估，轻量高效 |
| **Image2D** (`MazeBench-2D-Image`) | PNG 图像（绿色起点 `S`，红色终点 `G`，高清渲染） | 文本路径 | 多模态视觉语言模型评测 |

✅ **底层一致**：相同迷宫生成器、最短路径验证器、评分体系与 HTML/PDF 报告模块。

---

## 快速开始

```bash
# 1. 安装依赖（推荐 Python ≥3.10）
pip install -r requirements.txt

# 2. 运行完整评测（文本+图像，Mock 模型）
python bench.py

# 3. 查看可视化报告
open outputs/report_*.html
```

> 🔔 **无需 API Key** — 默认使用 `MockAdapter`（离线可用）。设置 `OPENAI_API_KEY` 即可启用 `gpt-4o`（支持图像输入）。

---

## 🛠️ 配置与运行逻辑

### 配置：`config/config.yaml`
```yaml
text2d:
  size: '11x11'         # 推荐奇数×奇数（11–41）
  algorithm: dfs        # dfs / prim / prim_loops
  seed: 123
  # 坑（traps）特性已移除；生成器和评测不再包含陷阱或 trap_zones 字段

image2d:
  size: '11x11'
  n: 5                  # 批量生成数量
  algorithm: prim       # dfs / prim / prim_loops
  seed: 123
  cell_px: 20           # 每格像素数（值越大图像越清晰）

  start_goal: random       # corner / random（随机起终点）

model: mock             # mock / gpt-4o / claude-3-5-sonnet
output_dir: outputs/
mode: all               # text2d / image2d / all
generate_only: false    # 仅生成迷宫，不评测
pre_generated_dir: ""   # 指定预生成迷宫目录
```

> 💡 可通过 `config/local.yaml`（已添加 `.gitignore`）或环境变量覆盖配置。

---

## 📁 输出目录结构（`outputs/`）
```
outputs/
├── mazes_text2d/          # 生成的 JSON 迷宫（generate_only 模式）
├── mazes_image2d/         # PNG + JSON 资产
├── text2d_summary.json    # 详细指标 + 汇总统计
├── image2d_summary.json
├── text2d_summary.pdf     # PDF 报告（ReportLab 生成）
├── image2d_summary.pdf
├── overview.json          # 跨模式平均分
├── report_text2d_*.html   # 交互式 HTML 报告（路径对比/热力图/失败样例）
└── report_image2d_*.html
```

---

## 🧩 核心特性

### 1. **结构优先迷宫生成**（`common/maze_generator.py`）
- 🌐 **摒弃“墙密度”参数**，生成更自然的布局  
- 🛣️ **主路径生成**：带局部转向偏置的随机游走（避免长直道/对角主导）  
- 🌳 **分支控制**：沿主路径采样，生成短分支+死胡同  
- 🧱 **墙体分散**：打破大块连续墙体，限制全局墙占比  
- 🕳️ 陷阱机制：历史版本存在；当前版本已移除该特性以简化评测（评测与生成均不再包含 traps 字段）

**内置算法**（支持插件式扩展）：
- `dfs`：步长为 2 的深度优先“挖墙法”，分支丰富，死胡同明显  
- `prim`：步长为 2 的随机 Prim 生长法，通道均匀
- `prim_loops`：先用 Prim 生成树，再在候选墙上按比例开口（约 15%），形成环路，提升探索难度


▶️ 新增算法仅需 **10 行以内**：实现 `carve` 逻辑 → `register_algo()` 注册即可全局生效。

---

### 2. **四维评分体系（S/Q/O/A）**
每道题独立评分（0 或 1，部分项为 [0,1] 实数），并依据迷宫尺寸动态加权：

| 指标 | 含义 | 评分逻辑 |
|------|------|----------|
| **S**（成功率） | 路径有效性 | 是否到达终点且不撞墙？✅=1，否则 0 |
| **Q**（质量）   | 最优性   | 路径长度是否等于最短路径？✅=1，否则 0 |
| **O**（重合度） | 覆盖比例 | 使用 F1（precision/recall）在“节点级别”衡量模型路径与最短路径的重合度；精确考虑交集规模与路径长度，范围 [0,1] |
| **A**（防作弊） | 合规性   | 通过输出沙盒检查（仅坐标字符、无注入/解释性文本）✅=1，否则 0 |

📈 综合得分 = `w_S·S + w_Q·Q + w_O·O + w_A·A`，最终以 **百分制** 展示。
> 目前默认权重（随尺寸变化但总体趋势一致）在 `Metrics` 中实现：示例值（可调）：
> - 小尺寸：`S=0.50, Q=0.25, O=0.23, A=0.02`
> - 中尺寸：`S=0.45, Q=0.30, O=0.23, A=0.02`
> - 大尺寸：`S=0.40, Q=0.35, O=0.23, A=0.02`

> 注：鲁棒性（R）暂时停用，后续将以独立实验展示。

#### 输入侧的轻微扰动（Anti-Cheat 解释）
- 在构造提示时会注入一个 `nonce`（随机或由种子派生），用于打破硬编码匹配与复用旧答案的可能性。
- 同时通过**输出沙盒**将模型回复约束为坐标字符集合，避免从解释性文本或指令注入中“泄漏”路径信息。

---

### 3. **防作弊与可复现设计**
- 🔐 **输入扰动**：在提示中注入 `nonce=seed`，防止硬编码或模式匹配  
- 🧪 **输出沙盒**：仅保留坐标字符（`[(),0-9]`），过滤说明/注释  
- 🧩 **种子锁定**：相同 `(size, seed, algo)` → 100% 相同迷宫 & 最短路径  
- 📦 **批量种子管理**：基种子 `s` → 第 `i` 个迷宫使用 `seed = s + i`

---

### 4. **多模态适配器**
| 适配器 | 依赖 | 输入格式 |
|--------|------|----------|
| `MockAdapter` | 无 | 生成合理但随机的路径（离线/CI 必备） |
| `OpenAIAdapter` | `OPENAI_API_KEY` | 文本 + 图像（Image2D），支持 `gpt-4o` 系列 |
| *(可扩展)* | — | 继承 `BaseAdapter` 接口即可 |

---

## 📦 代码结构

```
├── bench.py                 # 统一入口（推荐使用）
├── config/
│   ├── config.yaml          # 默认配置
│   └── local.yaml           # 本地覆盖（.gitignore 已忽略）
├── common/
│   ├── maze_generator.py    # 公共生成核心
│   └── config_loader.py     # 配置合并（YAML + 环境变量）
├── MazeBench-2D/           # 文本模式
│   ├── cli.py               # 独立运行入口
│   ├── maze_gen/
│   ├── eval_core/           # 解析器、验证器、评分器
│   └── report/
└── MazeBench-2D-Image/     # 图像模式
    ├── cli.py
    ├── maze_gen/            # + PNG 渲染器（cell_px、S/G 颜色）
    └── ...
```

---

## 🧪 高级用法

### 仅运行 Text2D
```bash
python MazeBench-2D/cli.py \
  --model gpt-4o \
  --size 21x21 \
  --algorithm prim \
  --start_goal random \
  --seed 42
```

### 评测预生成迷宫
```yaml
# config.yaml
generate_only: false
pre_generated_dir: "my_mazes/"
mode: text2d
```
→ 自动加载 `my_mazes/` 下的 `text2d_maze_*.json` 文件。

---

## 🧱 三维迷宫规划（Roadmap）

| 阶段 | 目标 |
|------|------|
| **A** | 3D 体素生成器（10³–40³），正交/等轴测渲染 |
| **B** | 6自由度路径验证（上下前后左右），最短路径计算 |
| **C** | 输入支持：图像堆栈 / 短视频；输出支持：指令序列 |
| **D** | 新指标：碰撞检测、步数预算、跨楼层、陷阱规避 |
| **E** | 统一多模态适配器（支持多帧输入） |
| **F** | CI 友好：Mock 3D 适配器 + 报告集成 |

🔧 *新增参数*：通道宽度、垂直通行率、分支频率、转向偏置、死胡同密度、陷阱比例等。

---

---

## 📜 开源协议

---

> 🕹️ **立即体验**：  
> `python bench.py --model mock --text2d.size 11x11 --image2d.n 3`  
> → 10 秒内生成完整报告（笔记本电脑即可运行）。

---

让我们携手推进空间推理评测——一次迷宫，一次进步 🧩

### 使用 OpenAI SDK 风格调用其他厂商（自定义 base_url + API Key 环境变量）

很多厂商（如火山引擎 Ark）兼容 OpenAI SDK/HTTP 接口。MazeBench 允许通过配置切换至这些厂商：

1) 在 `config/config.yaml`（或 `config/local.yaml`）中设置：
```yaml
OPENAI_API_BASE: 'https://ark.cn-beijing.volces.com/api/v3'  # 兼容 OpenAI 的基础 URL
OPENAI_API_KEY_ENV: 'ARK_API_KEY'                            # API Key 所在的环境变量名
USE_OPENAI_SDK: true                                         # 优先使用 OpenAI SDK；失败则自动回退 HTTP
```

2) 将厂商提供的 Key 写入环境变量：
```bash
export ARK_API_KEY="your_vendor_key_here"
```

3) 运行（bench 或独立 CLI 都支持）：
```bash
# 统一入口（bench）
python bench.py --model openai-gpt-4o

# 文本模式 CLI
python MazeBench-2D/cli.py --model openai-gpt-4o --size 21x21 --algorithm prim

# 图像模式 CLI
python MazeBench-2D-Image/cli.py --size 21x21 --algorithm dfs
```

说明：
- 如果未安装 `openai` 包或 SDK 调用失败，会自动回退到 HTTP `requests` 调用。
- 也可以通过环境变量直接设置：`OPENAI_API_BASE`、`OPENAI_API_KEY_ENV`、`USE_OPENAI_SDK=1`。
