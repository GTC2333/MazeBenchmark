import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from tqdm import tqdm

from maze_gen.generator import MazeConfig, MazeGenerator
from eval_core.parser import OutputParser
from eval_core.validator import Validator
from eval_core.metrics import Metrics
from report.generator import generate_report
from model_gateways.openai import OpenAIAdapter
from model_gateways.anthropic import AnthropicAdapter
from model_gateways.mock import MockAdapter
from config.anti_cheat_rules import AntiCheat


def build_prompt(maze):
    grid = maze['grid']
    start = maze['start']
    goal = maze['goal']
    return (
        f"迷宫大小 {len(grid)}x{len(grid[0])}. 起点{start}, 终点{goal}. "
        f"请输出纯坐标路径列表，如 [(0,0),(0,1),...]. 禁止解释。"
    )


def get_adapter(model_name: str):
    if model_name.startswith('mock'):
        return MockAdapter(model=model_name)
    if model_name.startswith('gpt') or model_name.startswith('openai'):
        import os
        if not os.environ.get('OPENAI_API_KEY'):
            return MockAdapter(model='mock-'+model_name)
        return OpenAIAdapter(model=model_name)
    if model_name.startswith('claude') or model_name.startswith('anthropic'):
        import os
        if not os.environ.get('ANTHROPIC_API_KEY'):
            return MockAdapter(model='mock-'+model_name)
        return AnthropicAdapter(model=model_name)
    raise ValueError('unknown model')


def run_single(size: str, model: str, outdir: Path, start_goal: str, algorithm: str):
    h, w = map(int, size.split('x'))
    cfg = MazeConfig(width=w, height=h, start_goal=start_goal, algorithm=algorithm)
    gen = MazeGenerator(cfg)
    maze = gen.generate()
    anti = AntiCheat(seed=maze.get('nonce', 0))
    maze_p = anti.perturb_input(maze)
    prompt = build_prompt(maze_p)
    adapter = get_adapter(model)
    text = adapter.generate(prompt)
    text = anti.sandbox_output(text)
    parser = OutputParser()
    parsed = parser.parse_with_fallback(text, adapter=adapter, prompt="请只输出坐标路径列表，如 [(0,0),(0,1),...]。")
    validator = Validator(maze['grid'], maze['start'], maze['goal'], maze['trap_zones'], maze['shortest_path'])
    result = validator.validate(parsed.path)
    metrics = Metrics(size=max(h, w))
    scores = metrics.score(result)
    failure_snapshot = '' if result.get('ok') else result.get('error', '')
    report_path = outdir / f"report_{model}_{h}x{w}.html"
    generate_report(str(report_path), maze, parsed.path, scores, failure_snapshot)
    return {'scores': scores, 'path_mode': parsed.mode, 'report': str(report_path)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--size', default='10x10')
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--outdir', default='examples')
    ap.add_argument('--start_goal', choices=['corner','random'], default='corner', help='起点/终点放置策略')
    ap.add_argument('--algorithm', choices=['dfs','prim'], default='dfs', help='迷宫生成算法')
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    sizes = [s.strip() for s in args.size.split(',')]

    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(run_single, s, args.model, outdir, args.start_goal, args.algorithm): s for s in sizes}
        for f in tqdm(as_completed(futs), total=len(futs)):
            try:
                results.append(f.result())
            except Exception as e:
                results.append({'error': str(e)})
    # Save summary
    (outdir / 'summary.json').write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(results, ensure_ascii=False))

if __name__ == '__main__':
    main()
