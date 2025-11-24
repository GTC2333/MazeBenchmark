from dataclasses import dataclass
from typing import Dict, Optional
import numpy as np
from .traps import TrapInjector
from common.maze_generator import CommonMazeConfig, CommonMazeGenerator

@dataclass
class MazeConfig:
    width: int
    height: int
    trap_ratio: float = 0.2
    seed: Optional[int] = None
    start_goal: str = 'corner'  # 'corner' or 'random'
    algorithm: str = 'dfs'  # 'dfs' or 'prim'

class MazeGenerator:
    def __init__(self, cfg: MazeConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.traps = TrapInjector(self.rng)
        self.core = CommonMazeGenerator(CommonMazeConfig(width=cfg.width, height=cfg.height, seed=cfg.seed, start_goal=self.cfg.start_goal, algo=self.cfg.algorithm))

    def generate(self) -> Dict:
        base = self.core.generate()
        grid = np.array(base['grid'], dtype=np.int8)
        trap_zones = self.traps.inject(grid, ratio=self.cfg.trap_ratio)
        sp = self.core._shortest_path(grid, base['start'], base['goal'])
        return {
            'width': base['width'],
            'height': base['height'],
            'grid': grid.tolist(),
            'start': base['start'],
            'goal': base['goal'],
            'trap_zones': trap_zones,
            'shortest_path': sp,
        }

