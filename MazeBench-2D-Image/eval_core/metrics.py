class Metrics:
    def __init__(self, size: int = 10):
        # Dynamic weights similar to text 2D
        self.wS = 0.4
        self.wQ = 0.3
        self.wR = 0.2
        self.wA = 0.1

    def score(self, result: dict) -> dict:
        S = 1 if result.get('ok') else 0
        Q = result.get('optimal', 0)
        R = 1 if result.get('ok') else 0
        A = 1
        total = round(100 * (self.wS*S + self.wQ*Q + self.wR*R + self.wA*A), 2)
        return {'S': S, 'Q': Q, 'R': R, 'A': A, 'total': total}
