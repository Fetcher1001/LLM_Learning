from importlib.metadata import version
import tiktoken
print("tiktoken version:", version("tiktoken"))
tokenizer = tiktoken.get_encoding("gpt2")
text = (
"Hello, do you like tea <|endoftext|> In the sunlit terraces"
"of someunknownPlace"
)
text1 = text + "<|endoftext|>" + "Akwirw ier"
integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
print("Encoded integers:", integers)
strings = tokenizer.decode(integers)
print("Decoded strings:", strings)

integers1 = tokenizer.encode(text1, allowed_special={"<|endoftext|>"})
print("Encoded integers1:", integers1)
strings1 = tokenizer.decode(integers1)
print("Decoded strings1:", strings1) 


path_to_file = "LLM_Learning/Tokenization_of_Txt/the-verdict.txt"
with open(path_to_file, "r", encoding="utf-8") as file:
    raw_text = file.read()

enc_text = tokenizer.encode(raw_text, allowed_special={"<|endoftext|>"})
print("Encoded integers from file:", enc_text)
print(len(enc_text))

enc_sample = enc_text[50:]

context_size = 4
x = enc_sample[:context_size]
y = enc_sample[1:context_size + 1]
print("x:", x)
print("y:", y)

for i in range(1, context_size + 1):
    context = enc_sample[:i]
    desired = enc_sample[i]
    print(tokenizer.decode(context), "---->", tokenizer.decode([desired]))