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

    def _wall_neighbors(self, r: int, c: int, grid: np.ndarray) -> List[Coord]:
        # 4-neighborhood, only walls
        res = []
        for dr, dc in [(1,0),(-1,0),(0,1),(0,-1)]:
            nr, nc = r+dr, c+dc
            if self._in_bounds(nr, nc, grid) and grid[nr, nc] == 1:
                res.append((nr, nc))
        return res

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

    def _biased_choice(self, r: int, c: int, goal: Coord, candidates: List[Coord], prev: Optional[Coord]) -> Optional[Coord]:
        if not candidates:
            return None
        weights = []
        md_now = abs(goal[0]-r) + abs(goal[1]-c)
        for nr, nc in candidates:
            md_next = abs(goal[0]-nr) + abs(goal[1]-nc)
            toward = 1.3 if md_next < md_now else 0.8
            if prev is None:
                turn = 1.0
            else:
                dr0, dc0 = prev[0]-r, prev[1]-c
                dr1, dc1 = nr-r, nc-c
                turn = 1.6 if (dr0, dc0) != (dr1, dc1) else 0.7
            weights.append(toward * turn)
        s = sum(weights)
        probs = [w/s for w in weights] if s > 0 else None
        idx = int(self.rng.choice(len(candidates), p=probs)) if probs else int(self.rng.integers(0, len(candidates)))
        return candidates[idx]

    def _carve_main_path(self, grid: np.ndarray, start: Coord, goal: Coord) -> List[Coord]:
        # Self-avoiding biased DFS: carve only into walls; backtrack when stuck
        stack: List[Coord] = [start]
        grid[start] = 0
        prev: Optional[Coord] = None
        visited: Set[Coord] = {start}
        h, w = grid.shape
        max_iters = h*w*8
        iters = 0
        while stack and stack[-1] != goal and iters < max_iters:
            iters += 1
            r, c = stack[-1]
            walls = self._wall_neighbors(r, c, grid)
            # Avoid creating 2x2 open squares: disallow carving a wall that would make 3 free neighbors
            pruned = []
            for nr, nc in walls:
                free_count = len(self._free_neighbors(nr, nc, grid))
                if free_count <= 1:  # conservative carving to keep tight corridors
                    pruned.append((nr, nc))
            if not pruned:
                # try original walls if pruned removed all
                pruned = walls
            nxt = self._biased_choice(r, c, goal, pruned, prev)
            if nxt is None:
                # backtrack
                stack.pop()
                prev = stack[-1] if stack else None
                continue
            nr, nc = nxt
            grid[nr, nc] = 0
            stack.append((nr, nc))
            prev = (nr, nc)
        return stack

    def _carve_branches(self, grid: np.ndarray, main_path: List[Coord]) -> Set[Coord]:
        carved: Set[Coord] = set(main_path)
        if len(main_path) < 4:
            return carved
        idxs = list(range(1, len(main_path)-1))
        self.rng.shuffle(idxs)
        take = max(1, int(0.15 * len(idxs)))  # fewer branches
        for k in idxs[:take]:
            r, c = main_path[k]
            n_br = int(self.rng.integers(1, 3))  # 1-2 branches
            for _ in range(n_br):
                br_len = int(self.rng.integers(2, 5))
                pr = None
                cr, cc = r, c
                for i in range(br_len):
                    walls = self._wall_neighbors(cr, cc, grid)
                    # prefer turning away from main path direction implicitly via bias
                    self.rng.shuffle(walls)
                    carved_step = False
                    for nr, nc in walls:
                        # stop branch if it would open into area with multiple free neighbors (keeps dead-end style)
                        if len(self._free_neighbors(nr, nc, grid)) > 1:
                            continue
                        grid[nr, nc] = 0
                        carved.add((nr, nc))
                        cr, cc = nr, nc
                        pr = (nr, nc)
                        carved_step = True
                        break
                    if not carved_step:
                        break
        return carved

    def _introduce_pillars(self, grid: np.ndarray, avoid: Set[Coord]) -> None:
        # Add walls in large open 3x3 blocks to prevent wide corridors; skip main/branch path cells
        h, w = grid.shape
        for r in range(h-2):
            for c in range(w-2):
                block = grid[r:r+3, c:c+3]
                if np.all(block == 0):
                    # place a pillar at center with some probability
                    if self.rng.random() < 0.35:
                        if (r+1, c+1) not in avoid:
                            block[1,1] = 1
    def _carve_prim_tree(self, grid: np.ndarray, start: Coord) -> None:
        # Randomized Prim-style growth: expand a spanning tree over the grid
        carved: Set[Coord] = set()
        grid[start] = 0
        carved.add(start)
        frontier: Set[Coord] = set(self._wall_neighbors(start[0], start[1], grid))
        h, w = grid.shape
        max_iters = h*w*8
        iters = 0
        while frontier and iters < max_iters:
            iters += 1
            # pick a random frontier wall cell
            idx = int(self.rng.integers(0, len(frontier)))
            cell = list(frontier)[idx]
            frontier.discard(cell)
            frn = self._free_neighbors(cell[0], cell[1], grid)
            # carve only if it connects to exactly one carved neighbor to keep tree structure
            if len(frn) == 1:
                r, c = cell
                grid[r, c] = 0
                carved.add(cell)
                for nbr in self._wall_neighbors(r, c, grid):
                    frontier.add(nbr)

    # Algorithm registry for extensibility
    def _apply_dfs(self, grid: np.ndarray, start: Coord, goal: Coord) -> None:
        main_path = self._carve_main_path(grid, start, goal)
        carved = self._carve_branches(grid, main_path)
        self._introduce_pillars(grid, carved)

    def _apply_prim(self, grid: np.ndarray, start: Coord, goal: Coord) -> None:
        self._carve_prim_tree(grid, start)

    def generate(self, start: Optional[Coord] = None, goal: Optional[Coord] = None) -> Dict:
        h, w = self.cfg.height, self.cfg.width
        if start is None or goal is None:
            if self.cfg.start_goal == 'random':
                start = (int(self.rng.integers(0, h)), int(self.rng.integers(0, w)))
                # ensure goal different
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
        # ensure endpoints
        grid[start] = 0
        grid[goal] = 0
        sp = self._shortest_path(grid, start, goal)
        if not sp:
            # emergency connect along zigzag
            cr, cc = start
            while (cr, cc) != goal:
                if cr != goal[0]:
                    step = int(np.sign(goal[0]-cr))
                    nr, nc = cr+step, cc
                else:
                    step = int(np.sign(goal[1]-cc))
                    nr, nc = cr, cc+step
                if self._in_bounds(nr, nc, grid):
                    grid[nr, nc] = 0
                    cr, cc = nr, nc
            sp = self._shortest_path(grid, start, goal)
        return {
            'width': w,
            'height': h,
            'grid': grid.tolist(),
            'start': start,
            'goal': goal,
            'shortest_path': sp,
        }
