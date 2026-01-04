from abc import ABC, abstractmethod

class SamplingStrategy(ABC):
    @abstractmethod
    def sample(self) -> float:
        pass
