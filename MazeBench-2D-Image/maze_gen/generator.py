from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import numpy as np
from PIL import Image, ImageDraw

Coord = Tuple[int, int]

@dataclass
class MazeConfig:
    width: int
    height: int
    density: float = 0.3
    trap_ratio: float = 0.2
    seed: Optional[int] = None
    cell_px: int = 24

class MazeGenerator:
    def __init__(self, cfg: MazeConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)

    def _neighbors(self, r: int, c: int, grid: np.ndarray) -> List[Coord]:
        h, w = grid.shape
        for dr, dc in [(1,0),(-1,0),(0,1),(0,-1)]:
            nr, nc = r+dr, c+dc
            if 0 <= nr < h and 0 <= nc < w and grid[nr, nc] == 0:
                yield (nr, nc)

    def _ensure_connectivity(self, grid: np.ndarray, start: Coord, goal: Coord) -> np.ndarray:
        from collections import deque
        def bfs_path():
            q = deque([start])
            prev = {start: None}
            while q:
                r,c = q.popleft()
                if (r,c) == goal:
                    break
                for nr,nc in self._neighbors(r,c,grid):
                    if (nr,nc) not in prev:
                        prev[(nr,nc)] = (r,c)
                        q.append((nr,nc))
            if goal not in prev:
                return None
            path = []
            cur = goal
            while cur is not None:
                path.append(cur)
                cur = prev[cur]
            return list(reversed(path))
        path = bfs_path()
        h, w = grid.shape
        attempts = 0
        while path is None and attempts < (h*w):
            attempts += 1
            r, c = start
            grid[r, c] = 0
            while (r, c) != goal:
                dr = int(np.sign((goal[0] - r)))
                dc = int(np.sign((goal[1] - c)))
                choices = [(r+dr, c), (r, c+dc)]
                if self.rng.random() < 0.5:
                    choices.reverse()
                nr, nc = choices[0]
                nr = min(max(nr, 0), h-1)
                nc = min(max(nc, 0), w-1)
                grid[nr, nc] = 0
                r, c = nr, nc
            path = bfs_path()
        return grid

    def _shortest_path(self, grid: np.ndarray, start: Coord, goal: Coord) -> List[Coord]:
        from collections import deque
        q = deque([start])
        prev = {start: None}
        while q:
            r, c = q.popleft()
            if (r, c) == goal:
                break
            for nr, nc in self._neighbors(r, c, grid):
                if (nr, nc) not in prev:
                    prev[(nr, nc)] = (r, c)
                    q.append((nr, nc))
        if goal not in prev:
            return []
        path = []
        cur = goal
        while cur is not None:
            path.append(cur)
            cur = prev[cur]
        return list(reversed(path))

    def generate(self) -> Dict:
        w, h = self.cfg.width, self.cfg.height
        grid = np.zeros((h, w), dtype=np.int8)
        wall_mask = self.rng.random((h, w)) < self.cfg.density
        grid[wall_mask] = 1
        start, goal = (0, 0), (h-1, w-1)
        grid[start] = 0
        grid[goal] = 0
        grid = self._ensure_connectivity(grid, start, goal)
        sp = self._shortest_path(grid, start, goal)
        return {
            'width': w,
            'height': h,
            'grid': grid.tolist(),
            'start': start,
            'goal': goal,
            'shortest_path': sp
        }

    def render_image(self, maze: Dict) -> Image.Image:
        cell = self.cfg.cell_px
        h, w = maze['height'], maze['width']
        img = Image.new('RGB', (w*cell, h*cell), (255,255,255))
        draw = ImageDraw.Draw(img)
        for r in range(h):
            for c in range(w):
                x0, y0 = c*cell, r*cell
                x1, y1 = x0+cell-1, y0+cell-1
                if maze['grid'][r][c] == 1:
                    draw.rectangle([x0, y0, x1, y1], fill=(0,0,0))
                else:
                    draw.rectangle([x0, y0, x1, y1], outline=(200,200,200))
        sx, sy = maze['start'][1]*cell, maze['start'][0]*cell
        gx, gy = maze['goal'][1]*cell, maze['goal'][0]*cell
        draw.rectangle([sx+2, sy+2, sx+cell-3, sy+cell-3], fill=(0,255,0))
        draw.rectangle([gx+2, gy+2, gx+cell-3, gy+cell-3], fill=(255,0,0))
        return img

if __name__ == '__main__':
    cfg = MazeConfig(width=10, height=10, density=0.3, trap_ratio=0.0, seed=42)
    gen = MazeGenerator(cfg)
    maze = gen.generate()
    img = gen.render_image(maze)
    img.save('maze_10x10.png')
