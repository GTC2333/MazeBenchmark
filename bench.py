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
    adapter = get_adapter(model, cfg.get('OPENAI_API_KEY'), image=True)

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
