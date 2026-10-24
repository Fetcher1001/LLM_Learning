import tiktoken

_tokenizer = tiktoken.get_encoding("gpt2")

def encode(text: str) -> list[int]:
    return _tokenizer.encode(text, allowed_special={"<|endoftext|>"})

def decode(token_ids: list[int]) -> str:
    return _tokenizer.decode(token_ids)