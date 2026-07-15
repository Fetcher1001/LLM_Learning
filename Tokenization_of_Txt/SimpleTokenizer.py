import re 
from fromgithub import get_text

text = get_text().get_text()

class SimpleTokenizer:
    def __init__(self, text=None):
        self.text = text
        self.unique_token = "<|UNC|>"
        self.end_token = "<|END_of_TEXT|>"
        self.vocab_size, self.vocab = self.__vocab__(self.tokenize_text(self.text))
        self.str_to_int = self.vocab
        self.int_to_str = {integer: token for token, integer in self.vocab.items()}
        # integer IDs for special tokens
        self.uncid = self.str_to_int[self.unique_token]
        self.endofline_id = self.str_to_int[self.end_token]


    def tokenize_text(self, text=None):
        """Tokenize into words, '--' as one token, and each other punctuation mark separately."""
        if text is None:
            text = self.text
        if text is None:
            return []


        pattern = re.compile(
            rf"{re.escape(self.unique_token)}|{re.escape(self.end_token)}|\w+|--|[^\w\s]",
            re.UNICODE,
        )
        tokens = pattern.findall(text)
        return tokens

    def __vocab__(self, tokens):
        unique_tokens = [self.unique_token, self.end_token] + sorted(set(tokens))
        vocab_size = len(unique_tokens)
        vocab = {token: integer for integer, token in enumerate(unique_tokens)}
        return vocab_size, vocab
    
    def encode(self, text):
        tokens = self.tokenize_text(text)
        ids = [self.str_to_int.get(token, self.uncid) for token in tokens]
        ids.append(self.endofline_id)
        return ids
    
    def decode(self, ids):
        tokens = []
        for idx, i in enumerate(ids):
            if i == self.endofline_id and idx == len(ids):
                continue
            token = self.int_to_str.get(i, self.unique_token)
            tokens.append(token)
        text = " ".join(tokens)
        text = re.sub(r"\s+([.,!?;:])", r"\1", text)
        return text

test = SimpleTokenizer(text)
test_text = """"It's the last he painted, you know, "Mrs. Gisburn said with pardonable pride."""
ids = test.encode(test_text)
print("Encoded IDs:", ids)
last = ids[-1]
print("Decoded Text:", test.decode(ids))
print(last)

text1 = "Hello, do you like tea"
text2 = "In the sunlit terraces of the palace"

print("\n-----------end_token/ unique_token: TESTING---------------\n")
text = f" {test.end_token} ".join((text1, text2))
ids = test.encode(text)
print("Encoded IDs for combined text:", ids)
decoded_text = test.decode(ids)
print("Decoded Text for combined text:", decoded_text)