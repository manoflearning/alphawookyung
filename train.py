"""
based on Andrej Karpathy's GPT lecture:
https://youtu.be/kCc8FmEb1nY?si=pHVr7LpfKvYyZ4Dj
"""
import torch
import os
from tokenizer import CharTokenizer
from model import GPT, GPTConfig

# hyperparameters
batch_size = 64
block_size = 256
n_layer = 6
n_head = 6
n_embd = 384
max_iters = 5000
eval_iters = 200
eval_interval = 500
learning_rate = 3e-4
dropout = 0.2
bias = False
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
# ---------------

torch.manual_seed(1337)

# open files
texts = []
def open_files(root_dir):
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.txt'):
                file_path = os.path.join(root, file)
                with open(file_path, 'r', encoding='utf-8') as f:
                    texts.append(f.read())
open_files('data/my-codes')

sorted(texts, key=lambda x: torch.rand(1))
text = '\n'.join(texts)

# tokenize
tok = CharTokenizer()
data = torch.tensor(tok.encode(text), dtype=torch.int64)

# train and test splits
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
model_args = dict(block_size=block_size, vocab_size=tok.vocab_size, n_layer=n_layer, n_head=n_head, n_embd=n_embd, dropout=dropout, bias=bias)
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

# save the model to disk
torch.save(model.state_dict(), 'model_weight/prototype-5.pth')