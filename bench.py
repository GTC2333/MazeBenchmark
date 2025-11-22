import json
from pathlib import Path
from typing import Dict
from tqdm import tqdm

from common.config_loader import load_config, apply_env_keys
from common.pdf_export import export_summary_pdf

# Import internals from both benches
from MazeBench-2D.maze_gen.generator import MazeConfig as TextMazeConfig, MazeGenerator as TextMazeGenerator
from MazeBench-2D.eval_core.parser import OutputParser as TextParser
from MazeBench-2D.eval_core.validator import Validator as TextValidator
from MazeBench-2D.eval_core.metrics import Metrics as TextMetrics
from MazeBench-2D.report.generator import generate_report as TextReport
from MazeBench-2D.model_gateways.openai import OpenAIAdapter as TextOpenAIAdapter
from MazeBench-2D.model_gateways.mock import MockAdapter as TextMockAdapter
from MazeBench-2D.config.anti_cheat_rules import AntiCheat as TextAntiCheat

from MazeBench-2D-Image.maze_gen.generator import MazeConfig as ImgMazeConfig, MazeGenerator as ImgMazeGenerator
from MazeBench-2D-Image.eval_core.parser import OutputParser as ImgParser
from MazeBench-2D-Image.eval_core.validator import Validator as ImgValidator
from MazeBench-2D-Image.eval_core.metrics import Metrics as ImgMetrics
from MazeBench-2D-Image.report.generator import generate_report as ImgReport
from MazeBench-2D-Image.model_gateways.openai import OpenAIAdapter as ImgOpenAIAdapter
from MazeBench-2D-Image.model_gateways.mock import MockAdapter as ImgMockAdapter
from MazeBench-2D-Image.config.anti_cheat_rules import AntiCheat as ImgAntiCheat


def get_adapter(model: str, openai_key: str | None) -> object:
    if model.startswith('mock') or not openai_key:
        return TextMockAdapter(model=model)  # same interface
    return TextOpenAIAdapter(model=model)


def run_text2d(cfg: Dict, outdir: Path) -> Dict:
    model = cfg.get('model', 'mock')
    size = (cfg.get('text2d', {}).get('size') or '10x10')
    h, w = map(int, size.split('x'))
    workers = int(cfg.get('text2d', {}).get('workers') or 4)
    adapter = get_adapter(model, cfg.get('OPENAI_API_KEY'))

    sizes = [size]
    results = []
    # sequential for simplicity and deterministic logs
    for s in tqdm(sizes, desc='Text2D'):
        cfg_m = TextMazeConfig(width=w, height=h)
        gen = TextMazeGenerator(cfg_m)
        maze = gen.generate()
        anti = TextAntiCheat(seed=maze.get('nonce', 0))
        maze_p = anti.perturb_input(maze)
        prompt = (
            f"迷宫大小 {len(maze_p['grid'])}x{len(maze_p['grid'][0])}. 起点{maze_p['start']}, 终点{maze_p['goal']}. "
            f"请输出纯坐标路径列表，如 [(0,0),(0,1),...]. 禁止解释。"
        )
        text = adapter.generate(prompt)
        text = anti.sandbox_output(text)
        parser = TextParser()
        parsed = parser.parse_with_fallback(text, adapter=adapter, prompt="请只输出坐标路径列表，如 [(0,0),(0,1),...]。")
        validator = TextValidator(maze['grid'], maze['start'], maze['goal'], maze['trap_zones'], maze['shortest_path'])
        result = validator.validate(parsed.path)
        metrics = TextMetrics(size=max(h, w))
        scores = metrics.score(result)
        failure_snapshot = '' if result.get('ok') else result.get('error', '')
        rpath = outdir / f"text2d_report_{model}_{h}x{w}.html"
        TextReport(str(rpath), maze, parsed.path, scores, failure_snapshot)
        results.append({'scores': scores, 'report': str(rpath)})
    summary_path = outdir / 'text2d_summary.json'
    summary = {'avg_total': round(sum(r['scores']['total'] for r in results)/len(results), 2), 'items': results}
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    export_summary_pdf(str(outdir / 'text2d_summary.pdf'), 'Text2D Summary', summary, image_paths=None)
    return summary


def run_image2d(cfg: Dict, outdir: Path) -> Dict:
    model = cfg.get('model', 'mock')
    size = (cfg.get('image2d', {}).get('size') or '10x10')
    h, w = map(int, size.split('x'))
    n = int(cfg.get('image2d', {}).get('n') or 3)
    adapter = get_adapter(model, cfg.get('OPENAI_API_KEY'))

    results = []
    img_paths = []
    for i in tqdm(range(n), desc='Image2D'):
        gen = ImgMazeGenerator(ImgMazeConfig(width=w, height=h, density=0.3, trap_ratio=0.0, seed=i, cell_px=24))
        maze = gen.generate()
        img = gen.render_image(maze)
        img_path = outdir / f"image2d_maze_{h}x{w}_{i}.png"
        img.save(img_path)
        img_paths.append(str(img_path))
        anti = ImgAntiCheat(seed=maze.get('nonce', 0))
        maze_p = anti.perturb_input(maze)
        prompt = f"请根据图片中的迷宫，从绿色起点到红色终点输出坐标路径列表。迷宫尺寸为 {h}x{w}。只输出[(r,c),...]，不要解释。"
        text = adapter.generate(prompt, image_path=str(img_path))
        text = anti.sandbox_output(text)
        parser = ImgParser()
        parsed = parser.parse_with_fallback(text, adapter=None)
        validator = ImgValidator(maze['grid'], maze['start'], maze['goal'], maze['shortest_path'])
        vres = validator.validate(parsed.path)
        scores = ImgMetrics().score(vres)
        fail = '' if vres.get('ok') else vres.get('error', '')
        rpath = outdir / f"image2d_report_{model}_{h}x{w}_{i}.html"
        ImgReport(str(rpath), maze, parsed.path, scores, fail, str(img_path))
        results.append({'scores': scores, 'report': str(rpath)})
    summary = {'avg_total': round(sum(r['scores']['total'] for r in results)/len(results), 2), 'items': results}
    (outdir / 'image2d_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    export_summary_pdf(str(outdir / 'image2d_summary.pdf'), 'Image2D Summary', summary, image_paths=img_paths)
    return summary


def main():
    cfg = load_config()
    apply_env_keys(cfg)
    outdir = Path(cfg.get('output_dir') or 'outputs')
    outdir.mkdir(parents=True, exist_ok=True)
    print('Running MazeBenchmark with model', cfg.get('model'))
    text_summary = run_text2d(cfg, outdir)
    img_summary = run_image2d(cfg, outdir)
    # overview meta
    overview = {
        'model': cfg.get('model'),
        'text2d_avg': text_summary['avg_total'],
        'image2d_avg': img_summary['avg_total']
    }
    (outdir / 'overview.json').write_text(json.dumps(overview, ensure_ascii=False, indent=2), encoding='utf-8')
    print('Done. Summaries saved to', outdir)

if __name__ == '__main__':
    main()
