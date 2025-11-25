from abc import ABC, abstractmethod

class Tokenizer(ABC):
    @abstractmethod
    def build_vocabulary_from_scenarios(self):
        # Basic whitespace tokenization
        pass

    @abstractmethod
    def encode(self, token):
        pass