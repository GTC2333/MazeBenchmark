# MazeBench-2D-Image

以图像为输入的二维迷宫评测。生成 PNG 迷宫（黑墙、白路、绿色起点、红色终点），提示文本要求模型返回坐标路径列表。

## 快速开始

```
pip install -r MazeBench-2D-Image/requirements.txt
python MazeBench-2D-Image/cli.py
open MazeBench-2D-Image/examples/report_10x10_0.html
```

## 流水线与模块
- maze_gen：生成迷宫并渲染为图像，保证起终点连通与最短路径输出
- eval_core：解析模型文本输出为坐标路径并验证合法性与最优性，计算 S/Q/R/A 评分
- model_gateways：MockAdapter 默认；设置 OPENAI_API_KEY 可启用 OpenAI 视觉接口
- report：生成 HTML 报告，嵌入迷宫图片与路径对比

## CI 友好
- 未配置外部密钥时自动使用 MockAdapter，端到端可运行并生成示例报告
- 产物保存在 examples/ 下（PNG 与 HTML）
