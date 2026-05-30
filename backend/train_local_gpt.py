"""
VSPL AI Platform — Custom Generative Pretrained Transformer (GPT) Training
File: backend/train_local_gpt.py
"""
import os
import sys
import io
import json
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

# Prevent UnicodeEncodeError on Windows standard output
if sys.platform.startswith('win') and (sys.stdout is None or sys.stdout.encoding != 'utf-8'):
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    except AttributeError:
        pass

# Paths
BASE = Path(__file__).parent
DATA_DIR = BASE / 'data'
MODEL_DIR = BASE / 'models'
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 65)
print(" 🤖 VSPL AI Platform — Local GPT LLM Training Suite")
print("=" * 65)

# ── 1. LOAD CORPUS AND CREATE VOCABULARY ──
print("\n[1/4] Loading VSPL Knowledge Base and compiling corpus...")
kb_path = DATA_DIR / 'knowledge_base.json'
if not kb_path.exists():
    print("   ⚠️ Knowledge base not found. Please verify file path.")
    sys.exit(1)

with open(kb_path, 'r', encoding='utf-8') as f:
    kb_data = json.load(f)

# Concatenate all topics and details to form training corpus
corpus_list = []
for item in kb_data:
    keywords = ", ".join(item['keywords'])
    corpus_list.append(f"Topic: {keywords}\nDetail: {item['answer']}\n\n")

corpus = "".join(corpus_list)
print(f"   ✅ Corpus loaded successfully. Total length: {len(corpus)} characters.")

# Create character-level vocabulary mapping
chars = sorted(list(set(corpus)))
vocab_size = len(chars)
print(f"   ✅ Vocabulary size: {vocab_size} unique characters.")

char_to_idx = {ch: i for i, ch in enumerate(chars)}
idx_to_char = {i: ch for i, ch in enumerate(chars)}

# Save vocabulary maps
vocab_assets = {
    'chars': chars,
    'char_to_idx': char_to_idx,
    'idx_to_char': idx_to_char,
    'vocab_size': vocab_size
}
with open(MODEL_DIR / 'local_gpt_vocab.pkl', 'wb') as f:
    pickle.dump(vocab_assets, f)
print(f"   ✅ Vocabulary mapping saved to: local_gpt_vocab.pkl")

# Convert corpus to PyTorch tensor indices
data = torch.tensor([char_to_idx[ch] for ch in corpus], dtype=torch.long)

# ── 2. DEFINE MINI-GPT TRANSFORMER MODEL ──
class Head(nn.Module):
    """ One head of self-attention with causal masking """
    def __init__(self, head_size, n_embd, block_size, dropout=0.2):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)   # (B, T, hs)
        q = self.query(x) # (B, T, hs)
        
        # Calculate attention scores ("affinities")
        wei = q @ k.transpose(-2, -1) * (C ** -0.5) # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # Causal masking
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        
        # Perform weighted aggregation of values
        v = self.value(x) # (B, T, hs)
        out = wei @ v    # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out

class MultiHeadAttention(nn.Module):
    """ Multiple heads of self-attention in parallel """
    def __init__(self, num_heads, head_size, n_embd, block_size, dropout=0.2):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, n_embd, block_size, dropout) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out

class FeedForward(nn.Module):
    """ Simple linear layers followed by a GELU non-linearity """
    def __init__(self, n_embd, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """
    def __init__(self, n_embd, n_head, block_size, dropout=0.2):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size, n_embd, block_size, dropout)
        self.ffwd = FeedForward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class MiniGPT(nn.Module):
    """ Decoder-only Generative Pretrained Transformer (GPT) """
    def __init__(self, vocab_size, n_embd=64, n_head=4, n_blocks=2, block_size=64, dropout=0.1):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head, block_size, dropout) for _ in range(n_blocks)])
        self.ln_f = nn.LayerNorm(n_embd) # Final layer normalization
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.block_size = block_size

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx) # (B, T, C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device)) # (T, C)
        x = tok_emb + pos_emb # (B, T, C)
        x = self.blocks(x)    # (B, T, C)
        x = self.ln_f(x)      # (B, T, C)
        logits = self.lm_head(x) # (B, T, vocab_size)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens, temperature=0.7, top_k=10):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 0.01)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# ── 3. TRAINING ROUTINE ──
print("\n[3/4] Initializing model & trainer parameters...")

# Hyperparameters
block_size = 64
n_embd = 64
n_head = 4
n_blocks = 2
dropout = 0.1
batch_size = 16
epochs = 300
learning_rate = 1e-3

# Dynamic command line arguments parser for Streamlit customization
if '--epochs' in sys.argv:
    try:
        epochs = int(sys.argv[sys.argv.index('--epochs') + 1])
    except Exception:
        pass

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MiniGPT(vocab_size=vocab_size, n_embd=n_embd, n_head=n_head, n_blocks=n_blocks, block_size=block_size, dropout=dropout)
model.to(device)
print(f"   👉 Model loaded to: {device} | Total parameters: {sum(p.numel() for p in model.parameters()):,}")

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

def get_batch(split_data, batch_size, block_size):
    ix = torch.randint(len(split_data) - block_size, (batch_size,))
    x = torch.stack([split_data[i:i+block_size] for i in ix])
    y = torch.stack([split_data[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

print(f"\n🚀 Training Mini-GPT Local LLM for {epochs} epochs...")

for epoch in range(1, epochs + 1):
    model.train()
    xb, yb = get_batch(data, batch_size, block_size)
    
    logits, loss = model(xb, yb)
    
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    
    if epoch % 30 == 0 or epoch == 1:
        print(f"   Epoch {epoch:03d}/{epochs:03d} | Cross-Entropy Training Loss: {loss.item():.4f}")

# Save weights
weights_path = MODEL_DIR / 'local_gpt.pth'
torch.save(model.state_dict(), weights_path)
print(f"\n[4/4] Saving Local GPT assets...")
print(f"   ✅ Saved local GPT weights to: {weights_path.name}")

# Quick generation test
print("\n📝 Running offline text generation sample:")
model.eval()
context = torch.zeros((1, 1), dtype=torch.long, device=device)
prompt = "Topic: grade\nDetail:"
encoded_prompt = torch.tensor([[char_to_idx[c] for c in prompt if c in char_to_idx]], dtype=torch.long, device=device)

with torch.no_grad():
    generated_indices = model.generate(encoded_prompt, max_new_tokens=150, temperature=0.6, top_k=5)
    sample_text = "".join([idx_to_char[idx.item()] for idx in generated_indices[0]])
    print("-" * 65)
    print(sample_text)
    print("-" * 65)

print("\n" + "=" * 65)
print("  🎉 LOCAL GPT MODEL TRAINING SUCCESSFULLY COMPLETED!")
print("=" * 65)
