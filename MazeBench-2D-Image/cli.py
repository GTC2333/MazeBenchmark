import os
import json
import sys
from pathlib import Path
from typing import Dict
from dataclasses import dataclass
from tqdm import tqdm

import argparse

# Ensure repo root on sys.path for script execution
REPO_ROOT = str(Path(__file__).resolve().parents[1])
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from maze_gen.generator import MazeConfig, MazeGenerator
from eval_core.parser import OutputParser
from eval_core.validator import Validator
from eval_core.metrics import Metrics
from config.anti_cheat_rules import AntiCheat
from report.generator import generate_report
from common.model_gateway import make_adapter_from_cfg, ChatAdapter, MockAdapter

@dataclass
class RunConfig:
    width: int = 10
    height: int = 10
    seed: int | None = None
    cell_px: int = 24
    n: int = 3
    out_dir: str = 'MazeBench-2D-Image/examples'


def build_prompt(maze: Dict) -> str:
    h, w = maze['height'], maze['width']
    return f"请根据图片中的迷宫，从绿色起点到红色终点输出坐标路径列表。迷宫尺寸为 {h}x{w}。只输出[(r,c),...]，不要解释。"


def get_adapter(model: str = 'mock') -> ChatAdapter | MockAdapter:
    return make_adapter_from_cfg({'model': model}, image=True)


def run_single(cfg: RunConfig, idx: int, model: str = 'mock'):
    gen = MazeGenerator(MazeConfig(width=cfg.width, height=cfg.height, seed=(cfg.seed or 0)+idx, cell_px=cfg.cell_px))
    maze = gen.generate()
    img = gen.render_image(maze)
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    img_path = out_dir / f"maze_{cfg.width}x{cfg.height}_{idx}.png"
    img.save(img_path)

    anti = AntiCheat(seed=maze.get('nonce', 0))
    maze_in = anti.perturb_input(maze)
    adapter = get_adapter(model)
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', default='10x10')
    parser.add_argument('--n', type=int, default=3)
    parser.add_argument('--cell_px', type=int, default=24)
    parser.add_argument('--start_goal', choices=['corner','random'], default='corner')
    parser.add_argument('--algorithm', choices=['dfs','prim','prim_loops'], default='dfs')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--out_dir', default='MazeBench-2D-Image/examples')
    parser.add_argument('--model', default='mock')
    args = parser.parse_args()
    h, w = map(int, args.size.split('x'))
    rcfg = RunConfig(width=w, height=h, n=args.n, cell_px=args.cell_px, out_dir=args.out_dir, seed=args.seed)
    results = []
    for i in tqdm(range(rcfg.n)):
        # pass start_goal via generator config
        gen = MazeGenerator(MazeConfig(width=rcfg.width, height=rcfg.height, seed=(rcfg.seed or 0)+i, cell_px=rcfg.cell_px, start_goal=args.start_goal, algorithm=args.algorithm))
        maze = gen.generate()
        img = gen.render_image(maze)
        out_dir = Path(rcfg.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        img_path = out_dir / f"maze_{rcfg.width}x{rcfg.height}_{i}.png"
        img.save(img_path)
        anti = AntiCheat(seed=maze.get('nonce', (rcfg.seed or 0)))
        maze_in = anti.perturb_input(maze)
        adapter = get_adapter(args.model)
        prompt = build_prompt(maze_in)
        text = adapter.generate(prompt, image_path=str(img_path))
        parser_o = OutputParser()
        parsed = parser_o.parse_with_fallback(text, adapter=None)
        val = Validator(maze['grid'], maze['start'], maze['goal'], maze['shortest_path'])
        vres = val.validate(parsed.path)
        scores = Metrics().score(vres)
        fail = '' if vres.get('ok') else f"Failure: {vres.get('error')} Raw: {parsed.raw[:200]}"
        report_path = out_dir / f"report_{rcfg.width}x{rcfg.height}_{i}.html"
        generate_report(str(report_path), maze, parsed.path, scores, fail, str(img_path))
        results.append({'scores': scores, 'report': str(report_path), 'raw': parsed.raw})
    avg_total = round(sum(r['scores']['total'] for r in results)/len(results), 2) if results else 0
    summary = {'avg_total': avg_total, 'items': results}
    Path(rcfg.out_dir, 'summary_image.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print('Done. See', rcfg.out_dir)

if __name__ == '__main__':
    main()
