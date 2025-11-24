class Metrics:
    def __init__(self, size: int = 10):
        # Dynamic weights similar to text 2D
        self.wS = 0.5
        self.wQ = 0.2
        self.wO = 0.2
        self.wR = 0.08
        self.wA = 0.02

    def score(self, result: dict) -> dict:
        S = 1 if result.get('ok') else 0
        Q = 1 if result.get('optimal') else 0
        O = float(result.get('overlap') or 0.0) if result.get('ok') else 0.0
        R = 1 if result.get('robust') else 0
        A = 1 if result.get('anti_cheat_pass', True) else 0
        total = round(100 * (self.wS*S + self.wQ*Q + self.wO*O + self.wR*R + self.wA*A), 2)
        return {'S': S, 'Q': Q, 'O': O, 'R': R, 'A': A, 'total': total}
