from abc import ABC, abstractmethod

class Predictor(ABC):
    @abstractmethod
    def get_name(self) -> str:
        pass

    @abstractmethod
    def predict(self, image_file) -> dict:
        pass