"""
based on Andrej Karpathy's GPT lecture:
https://youtu.be/kCc8FmEb1nY?si=pHVr7LpfKvYyZ4Dj
"""
import torch
import tiktoken
from model import GPT, GPTConfig

# hyperparameters
batch_size = 64
block_size = 64
n_layer = 6
n_head = 6
n_embd = 384
max_iters = 5000
eval_iters = 200
eval_interval = 500
learning_rate = 3e-4
dropout = 0.1
bias = False
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
# ---------------

torch.manual_seed(1337)

with open('data/my-cp-codes.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)

# create a mapping from characters to integers
stoi = { ch:i for i, ch in enumerate(chars) }
itos = { i:ch for i, ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# openai tokenizer tiktoken: https://github.com/openai/tiktoken
# enc = tiktoken.encoding_for_model("gpt-2")

# train and test splits
data = torch.tensor(encode(text), dtype=torch.int64)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# instantiate the model
model_args = dict(block_size=block_size, vocab_size=vocab_size, n_layer=n_layer, n_head=n_head, n_embd=n_embd,
                  dropout=dropout, bias=bias)
gptconf = GPTConfig(**model_args)
model = GPT(gptconf)
m = model.to(device)

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)

for iter in range(max_iters):
    if iter % eval_iters == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

start_string = '#include <'
context = torch.tensor(encode(start_string), dtype=torch.int64, device=device).view((1, len(start_string)))
print(decode(model.generate(context, max_new_tokens=500)[0].tolist()))

# save the model to disk
torch.save(model.state_dict(), 'model1.pth')