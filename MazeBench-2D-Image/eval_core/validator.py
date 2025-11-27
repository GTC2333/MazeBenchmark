from typing import List, Tuple, Dict

Coord = Tuple[int, int]

class Validator:
    def __init__(self, grid: List[List[int]], start: Coord, goal: Coord, shortest_path: List[Coord]):
        self.grid = grid
        self.start = start
        self.goal = goal
        self.shortest = shortest_path

    def validate(self, path: List[Coord]) -> Dict:
        # Compute overlap (F1) regardless of validity
        overlap = self._overlap_f1(path)
        # Validity checks
        if not path:
            return {'ok': False, 'error': 'empty_path', 'overlap': overlap}
        if path[0] != self.start:
            return {'ok': False, 'error': 'bad_start', 'overlap': overlap}
        seen = set([path[0]])
        for i in range(1, len(path)):
            r0,c0 = path[i-1]
            r1,c1 = path[i]
            if c1 < 0 or c1 >=len(self.grid) or r1 < 0 or r1 >=len(self.grid):
                return {'ok': False, 'error': 'out of maze', 'overlap': overlap}
            print(self.grid)
            if abs(r0-r1) + abs(c0-c1) != 1:
                return {'ok': False, 'error': 'non_step', 'overlap': overlap}
            if self.grid[r1][c1] == 1:
                return {'ok': False, 'error': 'hit_wall', 'overlap': overlap}
            if (r1,c1) in seen:
                return {'ok': False, 'error': 'loop', 'overlap': overlap}
            seen.add((r1,c1))
        if path[-1] != self.goal:
            return {'ok': False, 'error': 'bad_goal', 'overlap': overlap}
        optimal = int(len(path) == len(self.shortest))
        return {'ok': True, 'optimal': optimal, 'overlap': overlap}

    def _overlap_f1(self, path: List[Coord]) -> float:
        if not path or not self.shortest:
            return 0.0
        sp = set(tuple(p) for p in self.shortest)
        inter = sum(1 for p in path if tuple(p) in sp)
        if inter == 0:
            return 0.0
        precision = inter / len(path)
        recall = inter / len(sp)
        return (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
