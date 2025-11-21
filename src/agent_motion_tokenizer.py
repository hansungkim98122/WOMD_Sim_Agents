from tokenizer import Tokenizer

class AgentMotionTokenizer(Tokenizer):
    def __init__(self, vocab_file):
        super().__init__(vocab_file)

    def tokenize(self, text):
        # Custom tokenization logic for agent motion
        tokens = text.replace('.', ' . ').replace('!', ' ! ').split()
        return [token for token in tokens if token in self.vocab]