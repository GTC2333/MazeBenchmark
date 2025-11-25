import json
from pathlib import Path
from typing import Dict
try:
    from tqdm import tqdm
except Exception:
    class _NoTqdm:
        def __init__(self, total=None, desc=None): pass
        def update(self, n=1): pass
        def close(self): pass
    def tqdm(iterable=None, total=None, desc=None):
        if iterable is not None:
            return iterable
        return _NoTqdm(total=total, desc=desc)
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from common.config_loader import load_config, apply_env_keys
from common.pdf_export import export_summary_pdf
from common.model_gateway import make_adapter_from_cfg

# Map hyphenated directories to importable alias packages, then import submodules normally
import sys, types
from importlib import import_module

base_dir = Path(__file__).parent.resolve()
# ensure repo root on sys.path so "common" and others are importable
if str(base_dir) not in sys.path:
    sys.path.insert(0, str(base_dir))
_alias_map = {
    'MazeBench_2D': base_dir / 'MazeBench-2D',
    'MazeBench_2D_Image': base_dir / 'MazeBench-2D-Image',
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
_txt_anticheat = import_module('MazeBench_2D.config.anti_cheat_rules')

TextMazeConfig = _txt_gen.MazeConfig
TextMazeGenerator = _txt_gen.MazeGenerator
TextParser = _txt_parser.OutputParser
TextValidator = _txt_validator.Validator
TextMetrics = _txt_metrics.Metrics
TextReport = _txt_report.generate_report
TextAntiCheat = _txt_anticheat.AntiCheat

# Lazy-load Image2D modules only when needed to avoid hard dependency on Pillow
ImgMazeConfig = None
ImgMazeGenerator = None
ImgParser = None
ImgValidator = None
ImgMetrics = None
ImgReport = None
ImgAntiCheat = None


def get_adapter(model: str, openai_key: str | None, image: bool = False, openai_base: str | None = None, openai_key_env: str | None = None, use_sdk: bool | None = None) -> object:
    # Delegate to unified common adapter builder; cfg is composed from env-prepared values
    cfg = {
        'model': model,
        'OPENAI_API_KEY': openai_key,
        'OPENAI_API_BASE': openai_base,
        'OPENAI_API_KEY_ENV': openai_key_env,
        'USE_OPENAI_SDK': use_sdk,
        # Allow azure through env/config if present
        'AZURE_OPENAI_API_KEY': os.getenv('AZURE_OPENAI_API_KEY'),
        'AZURE_OPENAI_ENDPOINT': os.getenv('AZURE_OPENAI_ENDPOINT'),
        'AZURE_OPENAI_DEPLOYMENT': os.getenv('AZURE_OPENAI_DEPLOYMENT'),
        'AZURE_OPENAI_API_VERSION': os.getenv('AZURE_OPENAI_API_VERSION'),
        'PROVIDER': os.getenv('PROVIDER')
    }
    return make_adapter_from_cfg(cfg, image=image)


def run_text2d(cfg: Dict, outdir: Path) -> Dict:
    model = cfg.get('model', 'mock')
    size = (cfg.get('text2d', {}).get('size') or '10x10')
    h, w = map(int, size.split('x'))
    adapter = get_adapter(model, cfg.get('OPENAI_API_KEY'), image=False, openai_base=cfg.get('OPENAI_API_BASE'), openai_key_env=cfg.get('OPENAI_API_KEY_ENV'), use_sdk=cfg.get('USE_OPENAI_SDK'))

    n = int(cfg.get('text2d', {}).get('n') or 1)
    start_goal = (cfg.get('text2d', {}).get('start_goal') or 'corner')
    algorithm = (cfg.get('text2d', {}).get('algorithm') or 'dfs')
    base_seed = cfg.get('text2d', {}).get('seed') or 0
    workers = int(cfg.get('text2d', {}).get('workers') or max(1, min(n, (os.cpu_count() or 4))))

    def _task(i: int):
        cfg_m = TextMazeConfig(width=w, height=h, seed=base_seed + i, start_goal=start_goal, algorithm=algorithm)
        gen = TextMazeGenerator(cfg_m)
        maze = gen.generate()
        anti = TextAntiCheat(seed=maze.get('nonce', 0))
        maze_p = anti.perturb_input(maze)
        prompt = (
            f"迷宫大小 {len(maze_p['grid'])}x{len(maze_p['grid'][0])}。迷宫为{maze_p['grid']}。0表示通路可以移动；1表示墙壁，不可穿过。原点在左上角，x 为列索引，y 为行索引。起点{maze_p['start']}，终点{maze_p['goal']}。"
            f"请只输出坐标路径列表，如 [(y0,x0),(y1,x1),...]，不要解释。"
        )   
        # print(prompt)


        text = adapter.generate(prompt)
        text = anti.sandbox_output(text)
        parser = TextParser()

        parsed = parser.parse_with_fallback(text, adapter=adapter, prompt="请只输出坐标路径列表，如 [(0,0),(0,1),...]。")
        validator = TextValidator(maze['grid'], maze['start'], maze['goal'], maze['shortest_path'])
        result = validator.validate(parsed.path)
        scores = TextMetrics(size=max(h, w)).score(result)
        failure_snapshot = '' if result.get('ok') else result.get('error', '')
        rpath = outdir / f"text2d_report_{model}_{h}x{w}_{i}.html"
        TextReport(str(rpath), maze, parsed.path, scores, failure_snapshot)
        return {'scores': scores, 'report': str(rpath)}

    results = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_task, i): i for i in range(n)}
        pbar = tqdm(total=n, desc='Text2D')
        for fut in as_completed(futures):
            try:
                results.append(fut.result())
            except Exception as e:
                results.append({'scores': {'total': 0, 'S': 0, 'Q': 0, 'O': 0, 'A': 0}, 'report': '', 'error': str(e)})
            pbar.update(1)
        pbar.close()

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
    adapter = get_adapter(model, cfg.get('OPENAI_API_KEY'), image=True, openai_base=cfg.get('OPENAI_API_BASE'), openai_key_env=cfg.get('OPENAI_API_KEY_ENV'), use_sdk=cfg.get('USE_OPENAI_SDK'))
    # Lazy import image modules on first use
    global ImgMazeConfig, ImgMazeGenerator, ImgParser, ImgValidator, ImgMetrics, ImgReport, ImgAntiCheat
    if ImgMazeGenerator is None:
        _img_gen = import_module('MazeBench_2D_Image.maze_gen.generator')
        _img_parser = import_module('MazeBench_2D_Image.eval_core.parser')
        _img_validator = import_module('MazeBench_2D_Image.eval_core.validator')
        _img_metrics = import_module('MazeBench_2D_Image.eval_core.metrics')
        _img_report = import_module('MazeBench_2D_Image.report.generator')
        _img_anticheat = import_module('MazeBench_2D_Image.config.anti_cheat_rules')
        ImgMazeConfig = _img_gen.MazeConfig
        ImgMazeGenerator = _img_gen.MazeGenerator
        ImgParser = _img_parser.OutputParser
        ImgValidator = _img_validator.Validator
        ImgMetrics = _img_metrics.Metrics
        ImgReport = _img_report.generate_report
        ImgAntiCheat = _img_anticheat.AntiCheat


    workers = int(cfg.get('image2d', {}).get('workers') or max(1, min(n, (os.cpu_count() or 4))))

    img_paths = []

    def _task(i: int):
        gen = ImgMazeGenerator(ImgMazeConfig(width=w, height=h, seed=(cfg.get('image2d', {}).get('seed') or 0)+i, cell_px=int(cfg.get('image2d', {}).get('cell_px') or 24), start_goal=(cfg.get('image2d', {}).get('start_goal') or 'corner'), algorithm=(cfg.get('image2d', {}).get('algorithm') or 'dfs')))
        maze = gen.generate()
        img = gen.render_image(maze)
        img_path = outdir / f"image2d_maze_{h}x{w}_{i}.png"
        img.save(img_path)
        anti = ImgAntiCheat(seed=maze.get('nonce', 0))
        maze_p = anti.perturb_input(maze)
        prompt = f"请根据图片中的迷宫，从绿色起点到红色终点输出坐标路径列表,白色单元格为路径，可以在上面移动；黑色单元格为墙壁，禁止穿越墙壁。迷宫尺寸为 {h}x{w}。原点(0,0)在左上角.x 为列索引，y 为行索引。只输出[(y1,x1),(y2,x2),...]，不要解释。"
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
        return {'scores': scores, 'report': str(rpath), 'img_path': str(img_path)}

    results = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_task, i): i for i in range(n)}
        pbar = tqdm(total=n, desc='Image2D')
        for fut in as_completed(futures):
            try:
                r = fut.result()
                results.append({'scores': r['scores'], 'report': r['report']})
                img_paths.append(r['img_path'])
            except Exception as e:
                results.append({'scores': {'total': 0, 'S': 0, 'Q': 0, 'O': 0, 'A': 0}, 'report': '', 'error': str(e)})
            pbar.update(1)
        pbar.close()

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
        base_seed = cfg.get('text2d', {}).get('seed') or 0
        for i in tqdm(range(count), desc='GenOnly Text2D'):
            gen = TextMazeGenerator(TextMazeConfig(width=w, height=h, seed=base_seed+i, start_goal=start_goal, algorithm=algorithm))
            maze = gen.generate()
            (outdir / f'text2d_maze_{h}x{w}_{i}.json').write_text(json.dumps(maze, ensure_ascii=False), encoding='utf-8')
    elif mode == 'image2d':
        h, w = map(int, (cfg.get('image2d', {}).get('size') or '10x10').split('x'))
        base_seed = cfg.get('image2d', {}).get('seed') or 0
        for i in tqdm(range(count), desc='GenOnly Image2D'):
            gen = ImgMazeGenerator(ImgMazeConfig(width=w, height=h, seed=base_seed+i, cell_px=int(cfg.get('image2d', {}).get('cell_px') or 24), start_goal=(cfg.get('image2d', {}).get('start_goal') or 'corner'), algorithm=(cfg.get('image2d', {}).get('algorithm') or 'dfs')))
            maze = gen.generate()
            img = gen.render_image(maze)
            img.save(outdir / f'image2d_maze_{h}x{w}_{i}.png')
            (outdir / f'image2d_maze_{h}x{w}_{i}.json').write_text(json.dumps(maze, ensure_ascii=False), encoding='utf-8')


def eval_from_pregenerated(cfg: Dict, mazes_dir: Path, outdir: Path, mode: str = 'image2d') -> Dict:
    outdir.mkdir(parents=True, exist_ok=True)
    model = cfg.get('model', 'mock')
    if mode == 'text2d':
        adapter = get_adapter(model, cfg.get('OPENAI_API_KEY'), image=False, openai_base=cfg.get('OPENAI_API_BASE'), openai_key_env=cfg.get('OPENAI_API_KEY_ENV'), use_sdk=cfg.get('USE_OPENAI_SDK'))
        workers = int(cfg.get('text2d', {}).get('workers') or max(1, os.cpu_count() or 4))
        items = list(sorted(mazes_dir.glob('text2d_maze_*.json')))

        def _task(idx_mp):
            idx, mp = idx_mp
            maze = json.loads(mp.read_text(encoding='utf-8'))
            anti = TextAntiCheat(seed=maze.get('nonce', 0))
            maze_p = anti.perturb_input(maze)
            prompt = (
               f"迷宫大小 {len(maze_p['grid'])}x{len(maze_p['grid'][0])}。迷宫为{maze_p['grid']}。0表示通路可以移动；1表示墙壁，不可穿过。原点在左上角，x 为列索引，y 为行索引。起点{maze_p['start']}，终点{maze_p['goal']}。"
            f"请只输出坐标路径列表，如 [(y0,x0),(y1,x1),...]，不要解释。"
            )
            text = adapter.generate(prompt)
            text = anti.sandbox_output(text)
            parsed = TextParser().parse_with_fallback(text, adapter=adapter, prompt="请只输出坐标路径列表，如 [(0,0),(0,1),...]。")
            v = TextValidator(maze['grid'], maze['start'], maze['goal'], maze['shortest_path'])
            vres = v.validate(parsed.path)
            scores = TextMetrics(size=max(maze['height'], maze['width'])).score(vres)
            failure_snapshot = '' if vres.get('ok') else vres.get('error', '')
            rpath = outdir / f"text2d_report_{model}_{maze['height']}x{maze['width']}_{idx}.html"
            TextReport(str(rpath), maze, parsed.path, scores, failure_snapshot)
            return {'maze': mp.name, 'scores': scores, 'report': str(rpath)}

        results = []
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = {ex.submit(_task, (idx, mp)): idx for idx, mp in enumerate(items)}
            pbar = tqdm(total=len(items), desc='Eval Text2D')
            for fut in as_completed(futures):
                try:
                    results.append(fut.result())
                except Exception as e:
                    results.append({'maze': '', 'scores': {'total': 0, 'S': 0, 'Q': 0, 'O': 0, 'A': 0}, 'report': '', 'error': str(e)})
                pbar.update(1)
            pbar.close()
        avg = round(sum(r['scores']['total'] for r in results)/len(results), 2) if results else 0
        summary = {'avg_total': avg, 'items': results}
        (outdir / 'text2d_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
        export_summary_pdf(str(outdir / 'text2d_summary.pdf'), 'Text2D Summary', summary, image_paths=None)
        return summary
    else:
        adapter = get_adapter(model, cfg.get('OPENAI_API_KEY'), image=True, openai_base=cfg.get('OPENAI_API_BASE'), openai_key_env=cfg.get('OPENAI_API_KEY_ENV'), use_sdk=cfg.get('USE_OPENAI_SDK'))
        workers = int(cfg.get('image2d', {}).get('workers') or max(1, os.cpu_count() or 4))
        items = list(sorted(mazes_dir.glob('image2d_maze_*.json')))

        def _task(idx_jp):
            idx, jp = idx_jp
            maze = json.loads(jp.read_text(encoding='utf-8'))
            base = jp.stem
            png = mazes_dir / (base + '.png')
            anti = ImgAntiCheat(seed=maze.get('nonce', 0))
            maze_p = anti.perturb_input(maze)
            prompt = f"请根据图片中的迷宫，从绿色起点到红色终点输出坐标路径列表,白色单元格为路径，可以在上面移动；黑色单元格为墙壁，禁止穿越。迷宫尺寸为 {maze['height']}x{maze['width']}。原点在左上角，x 为列索引，y 为行索引。只输出[(y1,x1),(y2,x2),...]，不要解释。"
            text = adapter.generate(prompt, image_path=str(png))
            text = anti.sandbox_output(text)
            parsed = ImgParser().parse_with_fallback(text, adapter=None)
            v = ImgValidator(maze['grid'], maze['start'], maze['goal'], maze['shortest_path'])
            vres = v.validate(parsed.path)
            scores = ImgMetrics().score(vres)
            fail = '' if vres.get('ok') else vres.get('error', '')
            rpath = outdir / f"image2d_report_{model}_{maze['height']}x{maze['width']}_{idx}.html"
            ImgReport(str(rpath), maze, parsed.path, scores, fail, str(png))
            return {'maze': jp.name, 'scores': scores, 'report': str(rpath)}

        results = []
        img_paths = []
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = {ex.submit(_task, (idx, jp)): idx for idx, jp in enumerate(items)}
            pbar = tqdm(total=len(items), desc='Eval Image2D')
            for fut in as_completed(futures):
                try:
                    r = fut.result()
                    results.append({'maze': r['maze'], 'scores': r['scores'], 'report': r['report']})
                    # Can't know image path here without jp; skip accumulating images in summary
                except Exception as e:
                    results.append({'maze': '', 'scores': {'total': 0, 'S': 0, 'Q': 0, 'O': 0, 'A': 0}, 'report': '', 'error': str(e)})
                pbar.update(1)
            pbar.close()
        avg = round(sum(r['scores']['total'] for r in results)/len(results), 2) if results else 0
        summary = {'avg_total': avg, 'items': results}
        (outdir / 'image2d_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
        export_summary_pdf(str(outdir / 'image2d_summary.pdf'), 'Image2D Summary', summary, image_paths=None)
        return summary


def main():
    cfg = load_config()
    apply_env_keys(cfg)
    outdir = Path(cfg.get('output_dir') or 'outputs')
    outdir.mkdir(parents=True, exist_ok=True)
    # Optional generate-only
    gen_only = cfg.get('generate_only', False)
    use_pregen = cfg.get('use_pregenerated', False)
    preg_dir = cfg.get('pre_generated_dir') or ''
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
    if use_pregen:
        # Auto resolve pre-generated directory
        m = mode or 'all'
        base_dir = Path(preg_dir) if preg_dir else outdir
        if m in ('text2d','all'):
            tdir = base_dir / 'mazes_text2d'
            if not tdir.exists():
                generate_mazes_to_dir(cfg, tdir, 'text2d', count=count)
            text_summary = eval_from_pregenerated(cfg, tdir, outdir, mode='text2d')
        else:
            text_summary = {'avg_total': 0}
        if m in ('image2d','all'):
            idir = base_dir / 'mazes_image2d'
            if not idir.exists():
                generate_mazes_to_dir(cfg, idir, 'image2d', count=count)
            img_summary = eval_from_pregenerated(cfg, idir, outdir, mode='image2d')
        else:
            img_summary = {'avg_total': 0}
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
