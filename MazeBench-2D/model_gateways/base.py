from abc import ABC, abstractmethod
from typing import Dict, Any

class ModelAdapter(ABC):
    @abstractmethod
    def generate(self, prompt: str) -> str:
        raise NotImplementedError

    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError
