import os
import json
from pathlib import Path
from typing import Dict
from dataclasses import dataclass
from tqdm import tqdm

from maze_gen.generator import MazeConfig, MazeGenerator
from eval_core.parser import OutputParser
from eval_core.validator import Validator
from eval_core.metrics import Metrics
from config.anti_cheat_rules import AntiCheat
from report.generator import generate_report
from model_gateways.base import ModelAdapter
from model_gateways.openai import OpenAIAdapter
from model_gateways.mock import MockAdapter

@dataclass
class RunConfig:
    width: int = 10
    height: int = 10
    density: float = 0.3
    trap_ratio: float = 0.0
    seed: int | None = None
    cell_px: int = 24
    n: int = 3
    out_dir: str = 'MazeBench-2D-Image/examples'


def build_prompt(maze: Dict) -> str:
    h, w = maze['height'], maze['width']
    return f"请根据图片中的迷宫，从绿色起点到红色终点输出坐标路径列表。迷宫尺寸为 {h}x{w}。只输出[(r,c),...]，不要解释。"


def get_adapter() -> ModelAdapter:
    if os.getenv('OPENAI_API_KEY'):
        return OpenAIAdapter()
    return MockAdapter()


def run_single(cfg: RunConfig, idx: int):
    gen = MazeGenerator(MazeConfig(width=cfg.width, height=cfg.height, density=cfg.density, trap_ratio=cfg.trap_ratio, seed=(cfg.seed or 0)+idx, cell_px=cfg.cell_px))
    maze = gen.generate()
    img = gen.render_image(maze)
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    img_path = out_dir / f"maze_{cfg.width}x{cfg.height}_{idx}.png"
    img.save(img_path)

    anti = AntiCheat(seed=(cfg.seed or 0))
    maze_in = anti.perturb_input(maze)
    adapter = get_adapter()
    prompt = build_prompt(maze_in)
    text = adapter.generate(prompt, image_path=str(img_path))
    parser = OutputParser()
    parsed = parser.parse_with_fallback(text, adapter=None)

    val = Validator(maze['grid'], maze['start'], maze['goal'], maze['shortest_path'])
    vres = val.validate(parsed.path)
    scores = Metrics().score(vres)

    fail = '' if vres.get('ok') else f"Failure: {vres.get('error')} Raw: {parsed.raw[:200]}"
    report_path = out_dir / f"report_{cfg.width}x{cfg.height}_{idx}.html"
    generate_report(str(report_path), maze, parsed.path, scores, fail, str(img_path))
    return {'scores': scores, 'report': str(report_path), 'raw': parsed.raw}


def main():
    rcfg = RunConfig()
    results = []
    for i in tqdm(range(rcfg.n)):
        results.append(run_single(rcfg, i))
    summary = {
        'avg_total': round(sum(r['scores']['total'] for r in results)/len(results), 2),
        'items': results
    }
    Path(rcfg.out_dir, 'summary_image.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print('Done. See', rcfg.out_dir)

if __name__ == '__main__':
    main()
