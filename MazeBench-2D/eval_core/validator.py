from typing import List, Tuple, Dict, Any
import numpy as np

Coord = Tuple[int, int]

class Validator:
    def __init__(self, grid: List[List[int]], start: Coord, goal: Coord, shortest_path: List[Coord]):
        self.grid = np.array(grid, dtype=np.int8)
        self.start = tuple(start)
        self.goal = tuple(goal)
        self.shortest_path = [tuple(p) for p in shortest_path]

    def validate(self, path: List[Coord]) -> Dict[str, Any]:
        # Layer 1: validity
        err = self._check_validity(path)
        if err:
            return {'ok': False, 'error': f'validity_failure: {err}'}
        # Layer 2: optimality
        optimal = len(path) == len(self.shortest_path)
        # Layer 3: overlap between model path and shortest_path
        overlap = self._overlap(path)
        # Layer 4: robustness (minor perturbations should still succeed)
        robust = self._robustness(path)
        return {'ok': True, 'optimal': optimal, 'overlap': overlap, 'robust': robust}

    def _check_validity(self, path: List[Coord]) -> str:
        if not path:
            return 'empty_path'
        if tuple(path[0]) != self.start:
            return 'wrong_start'
        if tuple(path[-1]) != self.goal:
            return 'wrong_goal'
        # consecutive steps must be 4-neighbors and not through walls
        for i in range(1, len(path)):
            r0, c0 = path[i-1]
            r1, c1 = path[i]
            if abs(r0-r1) + abs(c0-c1) != 1:
                return 'illegal_move'
            if self.grid[r1, c1] == 1:
                return 'wall_collision'
    def _overlap(self, path: List[Coord]) -> float:
        if not self.shortest_path:
            return 0.0
        sp = set(self.shortest_path)
        inter = sum(1 for p in path if tuple(p) in sp)
        union = len(sp.union(set(tuple(p) for p in path)))
        return inter/union if union else 0.0

    # traps removed end-to-end

    def _robustness(self, path: List[Coord]) -> bool:
        # Tiny jitter on intermediate steps: remove or duplicate steps and see if still legal to goal via local correction
        # Here we simply check that small local deviations can be corrected greedily
        for idx in range(1, len(path)-1, max(1, len(path)//10)):
            p = path[idx]
            r, c = p
            for dr, dc in [(1,0),(-1,0),(0,1),(0,-1)]:
                nr, nc = r+dr, c+dc
                if 0 <= nr < self.grid.shape[0] and 0 <= nc < self.grid.shape[1] and self.grid[nr, nc] == 0:
                    # Check if from neighbor we can reach goal via simple greedy descent to shortest_path vicinity
                    if self._greedy_to_goal((nr, nc)):
                        return True
        return False

    def _greedy_to_goal(self, start: Coord) -> bool:
        # Greedy walk: try to approach goal; bail after limited steps
        r, c = start
        steps = 0
        H, W = self.grid.shape
        while steps < (H*W):
            if (r, c) == self.goal:
                return True
            choices = []
            for dr, dc in [(1,0),(-1,0),(0,1),(0,-1)]:
                nr, nc = r+dr, c+dc
                if 0 <= nr < H and 0 <= nc < W and self.grid[nr, nc] == 0:
                    choices.append((nr, nc))
            if not choices:
                return False
            # choose that minimizes manhattan to goal
            nr, nc = min(choices, key=lambda x: abs(x[0]-self.goal[0])+abs(x[1]-self.goal[1]))
            r, c = nr, nc
            steps += 1
        return False
