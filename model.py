"""
based on Andrej Karpathy's GPT lecture:
https://youtu.be/kCc8FmEb1nY?si=pHVr7LpfKvYyZ4Dj
"""
import torch
import torch.nn as nn
from torch.nn import functional as F

from dataclasses import dataclass

class LayerNorm(nn.Module):

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias   = nn.Parameter(torch.zeros(ndim)) if bias else None
    
    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout   = nn.Dropout(config.dropout)
        self.resid_dropout  = nn.Dropout(config.dropout)
        # config parameters
        self.n_head     = config.n_head
        self.n_embd     = config.n_embd
        self.dropout    = config.dropout
        # flash attention (requires PyTorch >= 2.0)
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print('WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0')
            self.register_buffer('bias', torch.tril(torch.ones(config.block_size, config.block_size))
                .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.shape

        k, q, v = self.c_attn(x).chunk(3, dim=-1) # (B, T, C)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        if self.flash:
            # (B, nh, T, hs)
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            # manual implementation of attention
            # (B, nh, T, hs) @ (B, nh, hs, T) -> (B, nh, T, T)
            att = q @ k.transpose(-2, -1) * (C // self.n_head)**-0.5
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            # (B, nh, T, T) @ (B, nh, T, hs) -> (B, nh, T, hs)
            y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C) # (B, T, C)

        y = self.resid_dropout(self.c_proj(y))
        return y

class FeedForward(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc       = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu       = nn.GELU()
        self.c_proj     = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout    = nn.Dropout(config.dropout)
    
    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1   = LayerNorm(config.n_embd, bias=config.bias)
        self.attn   = CausalSelfAttention(config)
        self.ln_2   = LayerNorm(config.n_embd, bias=config.bias)
        self.ffwd   = FeedForward(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.ffwd(self.ln_2(x))
        return x

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte     = nn.Embedding(config.vocab_size, config.n_embd),
            wpe     = nn.Embedding(config.block_size, config.n_embd),
            drop    = nn.Dropout(config.dropout),
            h       = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f    = LayerNorm(config.n_embd, bias=config.bias)
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # https://paperswithcode.com/method/weight-tying
        self.transformer.wte.weight = self.lm_head.weight

        # init all weights
        self.apply(self.__init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 * (2 * config.n_layer)**-0.5)

        # report number of parameters
        print('number of parameters: %.2fM' % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def __init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.shape
        assert t <= self.config.block_size, f'Cannot forward sequence of length {t}, block size is only {self.config.block_size}'
        pos = torch.arange(0, t, dtype=torch.long, device=device)

        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_sample=1)
            idx = torch.cat((idx, idx_next), dim=1)
        
        return idx