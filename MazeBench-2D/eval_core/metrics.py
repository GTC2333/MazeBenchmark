from typing import Dict

class Metrics:
    def __init__(self, size: int):
        self.size = size
        # dynamic weights: larger mazes emphasize optimality and robustness
        if size <= 10:
            self.w = {'S': 0.5, 'Q': 0.2, 'O': 0.2, 'R': 0.08, 'A': 0.02}
        elif size <= 20:
            self.w = {'S': 0.5, 'Q': 0.2, 'O': 0.2, 'R': 0.08, 'A': 0.02}
        else:
            self.w = {'S': 0.5, 'Q': 0.2, 'O': 0.2, 'R': 0.08, 'A': 0.02}

    def score(self, result: Dict) -> Dict:
        # S: Success (1 if ok else 0)
        S = 1.0 if result.get('ok') else 0.0
        # Q: Optimality (1 if optimal else 0)
        Q = 1.0 if result.get('ok') and result.get('optimal') else 0.0
        # O: Overlap with shortest path (0-1)
        O = float(result.get('overlap') or 0.0) if result.get('ok') else 0.0
        # R: Robustness (bool to 1/0)
        R = 1.0 if result.get('robust') else 0.0
        # A: Anti-cheat adherence (reduced weight)
        A = 1.0 if result.get('anti_cheat_pass', True) else 0.0
        total = self.w['S']*S + self.w['Q']*Q + self.w['O']*O + self.w['R']*R + self.w['A']*A
        return {'S': S, 'Q': Q, 'O': O, 'R': R, 'A': A, 'total': round(total*100, 1)}
