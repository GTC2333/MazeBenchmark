from .base import ModelAdapter


class MockAdapter(ModelAdapter):
    """
    CI-friendly mock: returns a path-like string by echoing a simple stepwise
    route from (0,0) to goal using right/down moves. No maze peeking.
    """
    def __init__(self, model: str = 'mock'):
        self._model = model

    def generate(self, prompt: str) -> str:
        # Try to detect size from prompt text; default 10x10
        import re
        m = re.search(r"(\d+)x(\d+)", prompt)
        h, w = (10, 10)
        if m:
            h, w = int(m.group(1)), int(m.group(2))
        path = []
        x, y = 0, 0
        path.append((x, y))
        while y < w - 1:
            y += 1
            path.append((x, y))
        while x < h - 1:
            x += 1
            path.append((x, y))
        return str(path)
