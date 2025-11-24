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
                # 标记为“死路”陷阱，但不改动迷宫拓扑，避免破坏可达性
                traps['dead_ends'].append((r, c))
            elif t == 'bottleneck':
                # 标记为“瓶颈”陷阱，不改动网格；评估器会在路径经过时判定失败
                traps['bottlenecks'].append((r, c))
            else:
                # 动态障碍：仅做标记，评估器负责处罚
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
