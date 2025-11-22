from abc import ABC, abstractmethod
from typing import Dict, Any

class ModelAdapter(ABC):
    @abstractmethod
    def generate(self, prompt: str) -> str:
        raise NotImplementedError

    def name(self) -> str:
        return self.__class__.__name__
