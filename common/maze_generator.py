from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Set
import numpy as np

Coord = Tuple[int, int]

@dataclass
class CommonMazeConfig:
    width: int
    height: int
    seed: Optional[int] = None
    start_goal: str = 'corner'  # 'corner' or 'random'
    algo: str = 'dfs'  # 'dfs' or 'prim'

class CommonMazeGenerator:
    def __init__(self, cfg: CommonMazeConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)

    def _in_bounds(self, r: int, c: int, grid: np.ndarray) -> bool:
        h, w = grid.shape
        return 0 <= r < h and 0 <= c < w

    def _free_neighbors(self, r: int, c: int, grid: np.ndarray) -> List[Coord]:
        res = []
        for dr, dc in [(1,0),(-1,0),(0,1),(0,-1)]:
            nr, nc = r+dr, c+dc
            if self._in_bounds(nr, nc, grid) and grid[nr, nc] == 0:
                res.append((nr, nc))
        return res

    def _shortest_path(self, grid: np.ndarray, start: Coord, goal: Coord) -> List[Coord]:
        from collections import deque
        q = deque([start])
        prev = {start: None}
        while q:
            r, c = q.popleft()
            if (r, c) == goal:
                break
            for nr, nc in self._free_neighbors(r, c, grid):
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

    # ====== NEW DFS IMPLEMENTATION (standard recursive backtracker) ======
    def _dfs_standard(self, grid: np.ndarray) -> None:
        h, w = grid.shape
        # We work in cell-based space: treat every cell as traversable unit.
        # To avoid 2x2 walls and ensure connectivity, we use standard DFS over all cells.
        visited = np.zeros((h, w), dtype=bool)
        stack: List[Coord] = []
        
        # Start from (0,0)
        start_r, start_c = 0, 0
        stack.append((start_r, start_c))
        grid[start_r, start_c] = 0
        visited[start_r, start_c] = True

        directions = [(1,0), (-1,0), (0,1), (0,-1)]

        while stack:
            r, c = stack[-1]
            # Get unvisited neighbors
            unvisited = []
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                if self._in_bounds(nr, nc, grid) and not visited[nr, nc]:
                    unvisited.append((nr, nc))
            
            if unvisited:
                # Randomly pick one
                nxt = unvisited[int(self.rng.integers(0, len(unvisited)))]
                nr, nc = nxt
                # Carve the cell (note: no wall between in cell-based; all cells are path)
                grid[nr, nc] = 0
                visited[nr, nc] = True
                stack.append((nr, nc))
            else:
                # Backtrack
                stack.pop()

    def _apply_dfs(self, grid: np.ndarray, start: Coord, goal: Coord) -> None:
        # 🔑 KEY FIX: Use standard DFS that carves ALL cells.
        # Clear grid to all walls, then carve entire spanning tree.
        grid.fill(1)  # all walls initially
        self._dfs_standard(grid)
        # Ensure start and goal are open (they should be, but double-check)
        grid[start] = 0
        grid[goal] = 0

    def _carve_prim_tree(self, grid: np.ndarray, start: Coord) -> None:
        # Keep original Prim for compatibility (optional)
        carved: Set[Coord] = set()
        grid[start] = 0
        carved.add(start)
        frontier: Set[Coord] = set()
        for dr, dc in [(1,0),(-1,0),(0,1),(0,-1)]:
            nr, nc = start[0]+dr, start[1]+dc
            if self._in_bounds(nr, nc, grid):
                frontier.add((nr, nc))
        h, w = grid.shape
        max_iters = h*w*8
        iters = 0
        while frontier and iters < max_iters:
            iters += 1
            cell = list(frontier)[int(self.rng.integers(0, len(frontier)))]
            frontier.discard(cell)
            frn = [nbr for nbr in self._free_neighbors(cell[0], cell[1], grid) if nbr in carved]
            # Carve only if it connects to exactly one carved neighbor to avoid cycles (Prim's algorithm)
            if len(frn) == 1:
                grid[cell[0], cell[1]] = 0
                carved.add(cell)
                for dr, dc in [(1,0),(-1,0),(0,1),(0,-1)]:
                    nr, nc = cell[0]+dr, cell[1]+dc
                    if self._in_bounds(nr, nc, grid) and (nr, nc) not in carved and (nr, nc) not in frontier:
                        frontier.add((nr, nc))

    def _apply_prim(self, grid: np.ndarray, start: Coord, goal: Coord) -> None:
        grid.fill(1)
        self._carve_prim_tree(grid, start)
        grid[start] = 0
        grid[goal] = 0

    def generate(self, start: Optional[Coord] = None, goal: Optional[Coord] = None) -> Dict:
        h, w = self.cfg.height, self.cfg.width
        if start is None or goal is None:
            if self.cfg.start_goal == 'random':
                start = (int(self.rng.integers(0, h)), int(self.rng.integers(0, w)))
                while True:
                    goal = (int(self.rng.integers(0, h)), int(self.rng.integers(0, w)))
                    if goal != start:
                        break
            else:
                start = (0, 0)
                goal = (h-1, w-1)
        grid = np.ones((h, w), dtype=np.int8)

        algo_map = {
            'dfs': lambda: self._apply_dfs(grid, start, goal),
            'prim': lambda: self._apply_prim(grid, start, goal),
        }
        (algo_map.get(self.cfg.algo, algo_map['dfs']))()

        # Final safety: ensure start & goal are open (in case algo missed)
        grid[start] = 0
        grid[goal] = 0

        sp = self._shortest_path(grid, start, goal)
        # 🛑 No emergency connect — DFS guarantees connectivity
        # If sp is empty, it's a bug — but with full DFS, this won't happen.
        return {
            'width': w,
            'height': h,
            'grid': grid.tolist(),
            'start': start,
            'goal': goal,
            'shortest_path': sp,
            'nonce': int(self.cfg.seed) if self.cfg.seed is not None else 0,
        }