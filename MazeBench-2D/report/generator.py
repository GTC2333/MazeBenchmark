from typing import Dict, List, Tuple
import json
from pathlib import Path
import base64
from PIL import Image, ImageDraw

TEMPLATE_PATH = Path(__file__).parent / 'html_template.j2'

def _render(template: str, context: Dict[str, str]) -> str:
    out = template
    for k, v in context.items():
        out = out.replace(f"%%{k}%%", v)
    return out


def _render_maze_image(maze: Dict, cell_px: int = 24) -> str:
    h, w = maze['height'], maze['width']
    img = Image.new('RGB', (w*cell_px, h*cell_px), (255,255,255))
    draw = ImageDraw.Draw(img)
    for r in range(h):
        for c in range(w):
            x0, y0 = c*cell_px, r*cell_px
            x1, y1 = x0+cell_px-1, y0+cell_px-1
            if maze['grid'][r][c] == 1:
                draw.rectangle([x0, y0, x1, y1], fill=(0,0,0))
            else:
                draw.rectangle([x0, y0, x1, y1], outline=(200,200,200))
    sx, sy = maze['start'][1]*cell_px, maze['start'][0]*cell_px
    gx, gy = maze['goal'][1]*cell_px, maze['goal'][0]*cell_px
    draw.rectangle([sx+2, sy+2, sx+cell_px-3, sy+cell_px-3], fill=(0,255,0))
    draw.rectangle([gx+2, gy+2, gx+cell_px-3, gy+cell_px-3], fill=(255,0,0))
    import io
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    return f"data:image/png;base64,{b64}"


def generate_report(output_path: str, maze: Dict, model_path: List[Tuple[int,int]], scores: Dict, failure_snapshot: str):
    html = TEMPLATE_PATH.read_text(encoding='utf-8')
    img_src = _render_maze_image(maze)
    ctx = {
        'TOTAL': str(scores['total']),
        'S': str(scores['S']),
        'Q': str(scores['Q']),
        'O': str(scores.get('O', 0)),
        'R': str(scores['R']),
        'A': str(scores['A']),
        'SP': json.dumps(maze['shortest_path']),
        'MP': json.dumps(model_path),
        'FAIL': failure_snapshot,
        'IMG_SRC': img_src,
    }
    rendered = _render(html, ctx)
    Path(output_path).write_text(rendered, encoding='utf-8')
