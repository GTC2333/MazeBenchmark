from typing import Dict, List, Tuple
import json
from pathlib import Path

TEMPLATE_PATH = Path(__file__).parent / 'html_template.j2'

def _render(template: str, context: Dict[str, str]) -> str:
    out = template
    for k, v in context.items():
        out = out.replace(f"%%{k}%%", v)
    return out


def generate_report(output_path: str, maze: Dict, model_path: List[Tuple[int,int]], scores: Dict, failure_snapshot: str):
    html = TEMPLATE_PATH.read_text(encoding='utf-8')
    ctx = {
        'TOTAL': str(scores['total']),
        'S': str(scores['S']),
        'Q': str(scores['Q']),
        'R': str(scores['R']),
        'A': str(scores['A']),
        'GRID': json.dumps(maze['grid']),
        'SP': json.dumps(maze['shortest_path']),
        'MP': json.dumps(model_path),
        'FAIL': failure_snapshot,
    }
    rendered = _render(html, ctx)
    Path(output_path).write_text(rendered, encoding='utf-8')
