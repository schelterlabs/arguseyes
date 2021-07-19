from abc import ABC, abstractmethod


class Refinement(ABC):
    @abstractmethod
    def _compute(self, pipeline):
        raise NotImplementedError
