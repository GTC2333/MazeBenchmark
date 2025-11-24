from dataclasses import dataclass
from typing import Dict, Optional
import numpy as np
from PIL import Image, ImageDraw
from common.maze_generator import CommonMazeConfig, CommonMazeGenerator

@dataclass
class MazeConfig:
    width: int
    height: int
    seed: Optional[int] = None
    cell_px: int = 24
    start_goal: str = 'corner'  # 'corner' or 'random'
    algorithm: str = 'dfs'  # 'dfs' or 'prim'

class MazeGenerator:
    def __init__(self, cfg: MazeConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.core = CommonMazeGenerator(CommonMazeConfig(width=cfg.width, height=cfg.height, seed=cfg.seed, start_goal=self.cfg.start_goal, algo=self.cfg.algorithm))

    def generate(self) -> Dict:
        return self.core.generate()

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
    cfg = MazeConfig(width=10, height=10, seed=42, cell_px=24, start_goal='corner')
    gen = MazeGenerator(cfg)
    maze = gen.generate()
    img = gen.render_image(maze)
    img.save('maze_10x10.png')
