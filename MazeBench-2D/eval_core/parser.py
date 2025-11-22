import re
import json
from typing import List, Tuple, Optional
from dataclasses import dataclass

Coord = Tuple[int, int]

@dataclass
class ParseResult:
    path: List[Coord]
    raw: str
    mode: str

class OutputParser:
    def parse(self, text: str) -> ParseResult:
        raw = text.strip()
        # Try coordinate list: [(0,0),(0,1),...]
        m = re.findall(r"\((\s*\d+\s*),(\s*\d+\s*)\)", raw)
        if m:
            path = [(int(a), int(b)) for a, b in m]
            return ParseResult(path=path, raw=raw, mode='coords')
        # Try JSON array of arrays
        try:
            obj = json.loads(raw)
            if isinstance(obj, list) and obj and isinstance(obj[0], list):
                path = [(int(r), int(c)) for r, c in obj]
                return ParseResult(path=path, raw=raw, mode='json_list')
        except Exception:
            pass
        # Direction sequence: U/D/L/R possibly with spaces/commas
        dirs = re.findall(r"[UDLR]", raw, flags=re.IGNORECASE)
        if dirs:
            path = self._dirs_to_path(dirs)
            return ParseResult(path=path, raw=raw, mode='dirs')
        # Mixed annotations: lines with coordinates and comments
        lines = [l for l in raw.splitlines() if '(' in l and ')' in l]
        coords = []
        for ln in lines:
            m2 = re.findall(r"\((\s*\d+\s*),(\s*\d+\s*)\)", ln)
            for a, b in m2:
                coords.append((int(a), int(b)))
        if coords:
            return ParseResult(path=coords, raw=raw, mode='annotated')
        # Row/col confusion: detect if path reversed and correct if plausible
        m3 = re.findall(r"\[(\s*\d+\s*),(\s*\d+\s*)\]", raw)
        if m3:
            path = [(int(b), int(a)) for a, b in m3]
            return ParseResult(path=path, raw=raw, mode='bracket_swap')
        # Fallback: extract numbers and pair
        nums = [int(x) for x in re.findall(r"\d+", raw)]
        if len(nums) >= 4 and len(nums) % 2 == 0:
            path = [(nums[i], nums[i+1]) for i in range(0, len(nums), 2)]
            return ParseResult(path=path, raw=raw, mode='numbers')
        # Final fallback: empty
        return ParseResult(path=[], raw=raw, mode='empty')

    def _dirs_to_path(self, dirs: List[str]) -> List[Coord]:
        cur = (0, 0)
        path = [cur]
        for d in dirs:
            dd = d.upper()
            if dd == 'U':
                cur = (cur[0]-1, cur[1])
            elif dd == 'D':
                cur = (cur[0]+1, cur[1])
            elif dd == 'L':
                cur = (cur[0], cur[1]-1)
            elif dd == 'R':
                cur = (cur[0], cur[1]+1)
            path.append(cur)
        return path

    def parse_with_fallback(self, text: str, adapter=None, prompt: Optional[str]=None) -> ParseResult:
        res = self.parse(text)
        if res.mode != 'empty' or adapter is None:
            return res
        if prompt is None:
            prompt = "只输出坐标路径列表，如 [(0,0),(0,1),...]。"
        try:
            new_text = adapter.generate(prompt)
            safe = ''.join(ch for ch in new_text if ch.isdigit() or ch in '[],() ')
            return self.parse(safe)
        except Exception:
            return res

