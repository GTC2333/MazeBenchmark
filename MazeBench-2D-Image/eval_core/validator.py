from typing import List, Tuple, Dict

Coord = Tuple[int, int]

class Validator:
    def __init__(self, grid: List[List[int]], start: Coord, goal: Coord, shortest_path: List[Coord]):
        self.grid = grid
        self.start = start
        self.goal = goal
        self.shortest = shortest_path

    def _neighbors(self, r: int, c: int):
        h, w = len(self.grid), len(self.grid[0])
        for dr, dc in [(1,0),(-1,0),(0,1),(0,-1)]:
            nr, nc = r+dr, c+dc
            if 0 <= nr < h and 0 <= nc < w and self.grid[nr][nc] == 0:
                yield (nr, nc)

    def validate(self, path: List[Coord]) -> Dict:
        if not path:
            return {'ok': False, 'error': 'empty_path'}
        if path[0] != self.start:
            return {'ok': False, 'error': 'bad_start'}
        seen = set([path[0]])
        for i in range(1, len(path)):
            r0,c0 = path[i-1]
            r1,c1 = path[i]
            if abs(r0-r1) + abs(c0-c1) != 1:
                return {'ok': False, 'error': 'non_step'}
            if self.grid[r1][c1] == 1:
                return {'ok': False, 'error': 'hit_wall'}
            if (r1,c1) in seen:
                return {'ok': False, 'error': 'loop'}
            seen.add((r1,c1))
        if path[-1] != self.goal:
            return {'ok': False, 'error': 'bad_goal'}
        optimal = int(len(path) == len(self.shortest))
        # overlap
        sp = set(tuple(p) for p in self.shortest)
        inter = sum(1 for p in path if tuple(p) in sp)
        union = len(sp.union(set(tuple(p) for p in path)))
        overlap = (inter/union) if union else 0.0
        # simple robustness: check that small deviations still lead to a valid neighbor
        robust = 1 if any((nr,nc) in sp for r,c in path[1:-1] for nr,nc in self._neighbors(r,c)) else 0
        return {'ok': True, 'optimal': optimal, 'overlap': overlap, 'robust': robust}
