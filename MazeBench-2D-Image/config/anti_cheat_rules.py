class AntiCheat:
    def __init__(self, seed: int = 0):
        pass

    def perturb_input(self, maze: dict) -> dict:
        maze = dict(maze)
        maze['nonce'] = maze.get('nonce', 0)
        return maze

    def sandbox_output(self, text: str) -> str:
        return ''.join(ch for ch in text if ch.isdigit() or ch in '[],() ')
