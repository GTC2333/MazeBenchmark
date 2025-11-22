import json
from typing import List, Tuple, Dict, Optional
import numpy as np
from pydantic import BaseModel, Field, validator
from .traps import TrapInjector

Coord = Tuple[int, int]

class MazeConfig(BaseModel):
    width: int = Field(..., ge=5, le=40)
    height: int = Field(..., ge=5, le=40)
    density: float = Field(0.3, ge=0.0, le=0.6)
    trap_ratio: float = Field(0.2, ge=0.0, le=0.6)
    seed: Optional[int] = None

    @validator('density', 'trap_ratio')
    def _validate_probs(cls, v):
        return float(v)

class MazeGenerator:
    def __init__(self, cfg: MazeConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.traps = TrapInjector(self.rng)

    def generate(self) -> Dict:
        w, h = self.cfg.width, self.cfg.height
        grid = np.zeros((h, w), dtype=np.int8)  # 0 free, 1 wall
        # Place random walls by density but ensure start/end free
        wall_mask = self.rng.random((h, w)) < self.cfg.density
        grid[wall_mask] = 1
        start, goal = (0, 0), (h-1, w-1)
        grid[start] = 0
        grid[goal] = 0
        # Build graph and ensure connectivity by carving using BFS shortest path existence
        grid = self._ensure_connectivity(grid, start, goal)
        # Inject traps
        trap_zones = self.traps.inject(grid, ratio=self.cfg.trap_ratio)
        # Recompute shortest path strictly on free cells; ensure non-empty by final connect
        grid = self._ensure_connectivity(grid, start, goal)
        sp = self._shortest_path(grid, start, goal)
        return {
            'width': w,
            'height': h,
            'grid': grid.tolist(),
            'start': start,
            'goal': goal,
            'trap_zones': trap_zones,
            'shortest_path': sp
        }

    def _neighbors(self, r: int, c: int, grid: np.ndarray) -> List[Coord]:
        h, w = grid.shape
        for dr, dc in [(1,0),(-1,0),(0,1),(0,-1)]:
            nr, nc = r+dr, c+dc
            if 0 <= nr < h and 0 <= nc < w and grid[nr, nc] == 0:
                yield (nr, nc)

    def _ensure_connectivity(self, grid: np.ndarray, start: Coord, goal: Coord) -> np.ndarray:
        # Carve a path using randomized BFS until start-goal connected
        h, w = grid.shape
        # Temporarily treat all free cells as nodes
        def bfs_path():
            from collections import deque
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
            # reconstruct
            path = []
            cur = goal
            while cur is not None:
                path.append(cur)
                cur = prev[cur]
            return list(reversed(path))
        path = bfs_path()
        # If not connected, carve corridors along a random walk from start to goal
        attempts = 0
        while path is None and attempts < (h*w):
            attempts += 1
            r, c = start
            grid[r, c] = 0
            while (r, c) != goal:
                # random step toward goal bias
                dr = np.sign((goal[0] - r))
                dc = np.sign((goal[1] - c))
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
        # BFS shortest path strictly on free cells
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

if __name__ == '__main__':
    cfg = MazeConfig(width=10, height=10, density=0.3, trap_ratio=0.2, seed=42)
    gen = MazeGenerator(cfg)
    maze = gen.generate()
    print(json.dumps(maze)[:200])
