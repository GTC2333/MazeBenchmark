from typing import Dict

class Metrics:
    def __init__(self, size: int):
        self.size = size
        # dynamic weights: larger mazes emphasize optimality and robustness
        if size <= 10:
            self.w = {'S': 0.35, 'Q': 0.25, 'R': 0.2, 'A': 0.2}
        elif size <= 20:
            self.w = {'S': 0.3, 'Q': 0.25, 'R': 0.25, 'A': 0.2}
        else:
            self.w = {'S': 0.25, 'Q': 0.25, 'R': 0.3, 'A': 0.2}

    def score(self, result: Dict) -> Dict:
        # S: Success (1 if ok else 0)
        S = 1.0 if result.get('ok') else 0.0
        # Q: Optimality (1 if optimal else 0.5 if warning suboptimal else 0)
        if result.get('ok') and result.get('optimal'):
            Q = 1.0
        elif result.get('ok') and result.get('warning') == 'suboptimal_path':
            Q = 0.5
        else:
            Q = 0.0
        # R: Robustness (bool to 1/0)
        R = 1.0 if result.get('robust') else 0.0
        # A: Anti-cheat adherence (placeholder 1 if passed sandbox)
        A = 1.0 if result.get('anti_cheat_pass', True) else 0.0
        total = self.w['S']*S + self.w['Q']*Q + self.w['R']*R + self.w['A']*A
        return {'S': S, 'Q': Q, 'R': R, 'A': A, 'total': round(total*100, 1)}
