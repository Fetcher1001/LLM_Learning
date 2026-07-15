import urllib.request
from pathlib import Path
import re 


class get_text:
    def __init__(self):
        self.url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt"
    
    def get_text(self):
        script_dir = Path(__file__).resolve().parent
        file_path = script_dir / "the-verdict.txt"

        urllib.request.urlretrieve(get_text().url, file_path)

        text = ""
        with open(file_path, "r", encoding="utf-8") as file:
            text = file.read()
        return text
        print("Number of characters in the text:", len(text))
        print(text[:99])

#
#def tokenize_text(text):
#    """Tokenize into words, '--' as one token, and each other punctuation mark separately."""
#    tokens = re.findall(r"\w+|--|[^\w\s]", text, re.UNICODE)
#    return tokens
#
#def words(tokens):
#    return sorted(set(tokens))
#
#tokens = tokenize_text(text)
#all_words = words(tokens)
#print(f"Number of tokens in the text: {len(tokens)}")
#print("First 10 tokens:", tokens[:30])
#
#vocab_size = len(words(tokens))
#vocab = {token:integer for integer, token in enumerate(words(tokens))}
#for i, item in enumerate(vocab.items()):
#    print(item)
#    if i >= 100:
#        break
#    