import json
from pathlib import Path
from typing import Dict
from tqdm import tqdm

from common.config_loader import load_config, apply_env_keys
from common.pdf_export import export_summary_pdf

# Map hyphenated directories to importable alias packages, then import submodules normally
import sys, types
from importlib import import_module

_alias_map = {
    'MazeBench_2D': Path('MazeBench-2D'),
    'MazeBench_2D_Image': Path('MazeBench-2D-Image'),
}
for alias, dpath in _alias_map.items():
    pkg = types.ModuleType(alias)
    pkg.__path__ = [str(dpath)]
    sys.modules[alias] = pkg

# Load Text2D modules via alias package
_txt_gen = import_module('MazeBench_2D.maze_gen.generator')
_txt_parser = import_module('MazeBench_2D.eval_core.parser')
_txt_validator = import_module('MazeBench_2D.eval_core.validator')
_txt_metrics = import_module('MazeBench_2D.eval_core.metrics')
_txt_report = import_module('MazeBench_2D.report.generator')
_txt_openai = import_module('MazeBench_2D.model_gateways.openai')
_txt_mock = import_module('MazeBench_2D.model_gateways.mock')
_txt_anticheat = import_module('MazeBench_2D.config.anti_cheat_rules')

TextMazeConfig = _txt_gen.MazeConfig
TextMazeGenerator = _txt_gen.MazeGenerator
TextParser = _txt_parser.OutputParser
TextValidator = _txt_validator.Validator
TextMetrics = _txt_metrics.Metrics
TextReport = _txt_report.generate_report
TextOpenAIAdapter = _txt_openai.OpenAIAdapter
TextMockAdapter = _txt_mock.MockAdapter
TextAntiCheat = _txt_anticheat.AntiCheat

# Load Image2D modules via alias package
_img_gen = import_module('MazeBench_2D_Image.maze_gen.generator')
_img_parser = import_module('MazeBench_2D_Image.eval_core.parser')
_img_validator = import_module('MazeBench_2D_Image.eval_core.validator')
_img_metrics = import_module('MazeBench_2D_Image.eval_core.metrics')
_img_report = import_module('MazeBench_2D_Image.report.generator')
_img_openai = import_module('MazeBench_2D_Image.model_gateways.openai')
_img_mock = import_module('MazeBench_2D_Image.model_gateways.mock')
_img_anticheat = import_module('MazeBench_2D_Image.config.anti_cheat_rules')

ImgMazeConfig = _img_gen.MazeConfig
ImgMazeGenerator = _img_gen.MazeGenerator
ImgParser = _img_parser.OutputParser
ImgValidator = _img_validator.Validator
ImgMetrics = _img_metrics.Metrics
ImgReport = _img_report.generate_report
ImgOpenAIAdapter = _img_openai.OpenAIAdapter
ImgMockAdapter = _img_mock.MockAdapter
ImgAntiCheat = _img_anticheat.AntiCheat


def get_adapter(model: str, openai_key: str | None, image: bool = False) -> object:
    if model.startswith('mock') or not openai_key:
        return ImgMockAdapter(model=model) if image else TextMockAdapter(model=model)
    return ImgOpenAIAdapter(model=model) if image else TextOpenAIAdapter(model=model)


def run_text2d(cfg: Dict, outdir: Path) -> Dict:
    model = cfg.get('model', 'mock')
    size = (cfg.get('text2d', {}).get('size') or '10x10')
    h, w = map(int, size.split('x'))
    workers = int(cfg.get('text2d', {}).get('workers') or 4)
    adapter = get_adapter(model, cfg.get('OPENAI_API_KEY'), image=False)

    sizes = [size]
    results = []
    # sequential for simplicity and deterministic logs
    for s in tqdm(sizes, desc='Text2D'):
        start_goal = (cfg.get('text2d', {}).get('start_goal') or 'corner')
        algorithm = (cfg.get('text2d', {}).get('algorithm') or 'dfs')
        cfg_m = TextMazeConfig(width=w, height=h, start_goal=start_goal, algorithm=algorithm)
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
    avg = round(sum(r['scores']['total'] for r in results)/len(results), 2) if results else 0
    summary = {'avg_total': avg, 'items': results}
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    export_summary_pdf(str(outdir / 'text2d_summary.pdf'), 'Text2D Summary', summary, image_paths=None)
    return summary


def run_image2d(cfg: Dict, outdir: Path) -> Dict:
    model = cfg.get('model', 'mock')
    size = (cfg.get('image2d', {}).get('size') or '10x10')
    h, w = map(int, size.split('x'))
    n = int(cfg.get('image2d', {}).get('n') or 3)
    adapter = get_adapter(model, cfg.get('OPENAI_API_KEY'), image=True)

    results = []
    img_paths = []
    for i in tqdm(range(n), desc='Image2D'):
        gen = ImgMazeGenerator(ImgMazeConfig(width=w, height=h, seed=i, cell_px=int(cfg.get('image2d', {}).get('cell_px') or 24), start_goal=(cfg.get('image2d', {}).get('start_goal') or 'corner'), algorithm=(cfg.get('image2d', {}).get('algorithm') or 'dfs')))
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
    avg = round(sum(r['scores']['total'] for r in results)/len(results), 2) if results else 0
    summary = {'avg_total': avg, 'items': results}
    (outdir / 'image2d_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    export_summary_pdf(str(outdir / 'image2d_summary.pdf'), 'Image2D Summary', summary, image_paths=img_paths)
    return summary


# Utility: generate-only and pre-generated evaluation support
def generate_mazes_to_dir(cfg: Dict, outdir: Path, mode: str = 'text2d', count: int = 5) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    if mode == 'text2d':
        h, w = map(int, (cfg.get('text2d', {}).get('size') or '10x10').split('x'))
        start_goal = (cfg.get('text2d', {}).get('start_goal') or 'corner')
        algorithm = (cfg.get('text2d', {}).get('algorithm') or 'dfs')
        for i in tqdm(range(count), desc='GenOnly Text2D'):
            gen = TextMazeGenerator(TextMazeConfig(width=w, height=h, start_goal=start_goal, algorithm=algorithm))
            maze = gen.generate()
            (outdir / f'text2d_maze_{h}x{w}_{i}.json').write_text(json.dumps(maze, ensure_ascii=False), encoding='utf-8')
    elif mode == 'image2d':
        h, w = map(int, (cfg.get('image2d', {}).get('size') or '10x10').split('x'))
        for i in tqdm(range(count), desc='GenOnly Image2D'):
            gen = ImgMazeGenerator(ImgMazeConfig(width=w, height=h, seed=i, cell_px=int(cfg.get('image2d', {}).get('cell_px') or 24), start_goal=(cfg.get('image2d', {}).get('start_goal') or 'corner'), algorithm=(cfg.get('image2d', {}).get('algorithm') or 'dfs')))
            maze = gen.generate()
            img = gen.render_image(maze)
            img.save(outdir / f'image2d_maze_{h}x{w}_{i}.png')
            (outdir / f'image2d_maze_{h}x{w}_{i}.json').write_text(json.dumps(maze, ensure_ascii=False), encoding='utf-8')


def eval_from_pregenerated(cfg: Dict, mazes_dir: Path, outdir: Path, mode: str = 'image2d') -> Dict:
    outdir.mkdir(parents=True, exist_ok=True)
    model = cfg.get('model', 'mock')
    if mode == 'text2d':
        adapter = get_adapter(model, cfg.get('OPENAI_API_KEY'), image=False)
        results = []
        for mp in sorted(mazes_dir.glob('text2d_maze_*.json')):
            maze = json.loads(mp.read_text(encoding='utf-8'))
            anti = TextAntiCheat(seed=maze.get('nonce', 0))
            maze_p = anti.perturb_input(maze)
            prompt = (
                f"迷宫大小 {len(maze_p['grid'])}x{len(maze_p['grid'][0])}. 起点{maze_p['start']}, 终点{maze_p['goal']}. "
                f"请输出纯坐标路径列表，如 [(0,0),(0,1),...]. 禁止解释。"
            )
            text = adapter.generate(prompt)
            text = anti.sandbox_output(text)
            parsed = TextParser().parse_with_fallback(text, adapter=adapter, prompt="请只输出坐标路径列表，如 [(0,0),(0,1),...]。")
            v = TextValidator(maze['grid'], maze['start'], maze['goal'], maze.get('trap_zones', []), maze['shortest_path'])
            vres = v.validate(parsed.path)
            scores = TextMetrics(size=max(maze['height'], maze['width'])).score(vres)
            results.append({'maze': mp.name, 'scores': scores})
        avg = round(sum(r['scores']['total'] for r in results)/len(results), 2) if results else 0
        summary = {'avg_total': avg, 'items': results}
        (outdir / 'text2d_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
        export_summary_pdf(str(outdir / 'text2d_summary.pdf'), 'Text2D Summary', summary)
        return summary
    else:
        adapter = get_adapter(model, cfg.get('OPENAI_API_KEY'), image=True)
        results = []
        img_paths = []
        for jp in sorted(mazes_dir.glob('image2d_maze_*.json')):
            maze = json.loads(jp.read_text(encoding='utf-8'))
            base = jp.stem
            png = mazes_dir / (base + '.png')
            img_paths.append(str(png))
            anti = ImgAntiCheat(seed=maze.get('nonce', 0))
            maze_p = anti.perturb_input(maze)
            prompt = f"请根据图片中的迷宫，从绿色起点到红色终点输出坐标路径列表。迷宫尺寸为 {maze['height']}x{maze['width']}。只输出[(r,c),...]，不要解释。"
            text = adapter.generate(prompt, image_path=str(png))
            text = anti.sandbox_output(text)
            parsed = ImgParser().parse_with_fallback(text, adapter=None)
            v = ImgValidator(maze['grid'], maze['start'], maze['goal'], maze['shortest_path'])
            vres = v.validate(parsed.path)
            scores = ImgMetrics().score(vres)
            results.append({'maze': jp.name, 'scores': scores})
        avg = round(sum(r['scores']['total'] for r in results)/len(results), 2) if results else 0
        summary = {'avg_total': avg, 'items': results}
        (outdir / 'image2d_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
        export_summary_pdf(str(outdir / 'image2d_summary.pdf'), 'Image2D Summary', summary, image_paths=img_paths)
        return summary


def main():
    cfg = load_config()
    apply_env_keys(cfg)
    outdir = Path(cfg.get('output_dir') or 'outputs')
    outdir.mkdir(parents=True, exist_ok=True)
    # Optional generate-only
    gen_only = cfg.get('generate_only', False)
    preg_dir = cfg.get('pre_generated_dir')
    mode = cfg.get('mode')  # text2d/image2d/all
    count = int(cfg.get('count') or cfg.get('image2d', {}).get('n') or 3)
    if gen_only:
        m = mode or 'all'
        if m in ('text2d','all'):
            generate_mazes_to_dir(cfg, outdir / 'mazes_text2d', 'text2d', count=count)
        if m in ('image2d','all'):
            generate_mazes_to_dir(cfg, outdir / 'mazes_image2d', 'image2d', count=count)
        print('Generate-only complete at', outdir)
        return
    # Optional evaluation from pre-generated assets
    if preg_dir:
        m = mode or 'image2d'
        if m == 'text2d':
            text_summary = eval_from_pregenerated(cfg, Path(preg_dir), outdir, mode='text2d')
            img_summary = {'avg_total': 0}
        elif m == 'image2d':
            img_summary = eval_from_pregenerated(cfg, Path(preg_dir), outdir, mode='image2d')
            text_summary = {'avg_total': 0}
        else:
            text_summary = eval_from_pregenerated(cfg, Path(preg_dir), outdir, mode='text2d')
            img_summary = eval_from_pregenerated(cfg, Path(preg_dir), outdir, mode='image2d')
    else:
        print('Running MazeBenchmark with model', cfg.get('model'))
        text_summary = run_text2d(cfg, outdir)
        img_summary = run_image2d(cfg, outdir)
    overview = {
        'model': cfg.get('model'),
        'text2d_avg': text_summary.get('avg_total', 0),
        'image2d_avg': img_summary.get('avg_total', 0)
    }
    (outdir / 'overview.json').write_text(json.dumps(overview, ensure_ascii=False, indent=2), encoding='utf-8')
    print('Done. Summaries saved to', outdir)

if __name__ == '__main__':
    main()
