from abc import ABC, abstractmethod

class ModelAdapter(ABC):
    @abstractmethod
    def generate(self, prompt: str, image_path: str | None = None) -> str:
        raise NotImplementedError

    def name(self) -> str:
        return self.__class__.__name__
