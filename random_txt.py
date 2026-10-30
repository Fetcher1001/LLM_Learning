import My_Gpt.tokenizer as tokenizer
import My_Gpt.model as model
import torch 

def generate_bs_txt(model, idx, max_new_tokens, context_length):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_length:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]
        probs = torch.softmax(logits, dim=-1)
        idx_next = torch.argmax(probs, dim=-1, keepdim=True)
        idx = torch.cat((idx, idx_next), dim=-1)
    return idx

starting_text = "Hello, my name is"
encoded_input = tokenizer.encode_ordinary(starting_text)
input_tensor = torch.tensor(encoded_input, dtype=torch.long).unsqueeze(0)
print("Input tensor shape:", input_tensor.shape)
print("encoded input:", encoded_input)

model = model.ShiGpt(model.GPT_CONFIG_124M)
model.eval()

out = generate_bs_txt(model, idx =input_tensor, max_new_tokens=6, context_length=model.GPT_CONFIG_124M["context_length"])
print("Output tensor shape:", out.shape)
decoded_output = tokenizer.decode(out.squeeze().tolist())
print("Decoded output:", decoded_output)