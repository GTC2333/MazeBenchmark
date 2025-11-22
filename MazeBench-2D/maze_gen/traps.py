from typing import List, Tuple, Dict
import numpy as np

Coord = Tuple[int, int]

class TrapInjector:
    def __init__(self, rng: np.random.Generator):
        self.rng = rng

    def inject(self, grid: np.ndarray, ratio: float) -> Dict[str, List[Coord]]:
        h, w = grid.shape
        traps = {
            'dead_ends': [],
            'bottlenecks': [],
            'dynamic_obstacles': []
        }
        # Identify candidate free cells
        free_cells = [(r, c) for r in range(h) for c in range(w) if grid[r, c] == 0]
        k = max(1, int(len(free_cells) * ratio))
        self.rng.shuffle(free_cells)
        candidates = free_cells[:k]
        for r, c in candidates:
            # Randomly assign trap type
            t = self.rng.choice(['dead_end', 'bottleneck', 'dynamic'])
            if t == 'dead_end':
                # Create a cul-de-sac by turning neighbors into walls except one
                neigh = list(self._neighbors(r, c, grid))
                self.rng.shuffle(neigh)
                for nr, nc in neigh[1:]:
                    grid[nr, nc] = 1
                traps['dead_ends'].append((r, c))
            elif t == 'bottleneck':
                # Narrow passage: set surrounding ring to walls leaving a single pass
                for nr, nc in self._ring(r, c, grid):
                    grid[nr, nc] = 1
                traps['bottlenecks'].append((r, c))
            else:
                # Dynamic: mark but keep cell free; evaluator can penalize stepping after t
                traps['dynamic_obstacles'].append((r, c))
        return traps

    def _neighbors(self, r: int, c: int, grid: np.ndarray):
        h, w = grid.shape
        for dr, dc in [(1,0),(-1,0),(0,1),(0,-1)]:
            nr, nc = r+dr, c+dc
            if 0 <= nr < h and 0 <= nc < w:
                yield (nr, nc)

    def _ring(self, r: int, c: int, grid: np.ndarray):
        h, w = grid.shape
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r+dr, c+dc
                if 0 <= nr < h and 0 <= nc < w:
                    yield (nr, nc)
