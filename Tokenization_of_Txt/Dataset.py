import tiktoken

class Dataset():
    def __init__(self, txt, max_length, stride):
        self.txt = txt
        self.max_length = max_length
        self.stride = stride 
        self.pairs = self.sliding_window()

    def sliding_window(self, offset = 1):
            tokens = self._tokenizer().encode(self.txt)
            length_tokens = len(tokens)
            input_ids = []
            target_ids = []

            for i in range(0, length_tokens, self.stride):
                if i + self.max_length >= length_tokens:
                    break
                input_ids.append(tokens[i, self.max_length])
                target_ids.append(tokens[i+offset : self.max_length + offset])

            return input_ids, target_ids

    def _tokenizer(self):
        return tiktoken.get_encoding("gpt2")
