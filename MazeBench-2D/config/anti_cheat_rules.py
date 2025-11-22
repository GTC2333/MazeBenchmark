from typing import Dict, List
import numpy as np

class AntiCheat:
    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def perturb_input(self, maze: Dict) -> Dict:
        # Shuffle labels of directions in prompt randomly, the parser is agnostic
        maze = dict(maze)
        maze['nonce'] = int(self.rng.integers(0, 1_000_000))
        return maze

    def sandbox_output(self, text: str) -> str:
        # Strip any non-numeric characters except brackets and commas
        return ''.join(ch for ch in text if ch.isdigit() or ch in '[],() ')
