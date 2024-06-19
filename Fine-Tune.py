import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.nn.utils.prune as prune
import mmap
import random
import pickle
from pathlib import Path
import pandas as pd
from pytorch_lamb import Lamb
import pyarrow.feather as feather
import re

# Check if CUDA is available and set the device accordingly

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.empty_cache()


block_size = 192
batch_size = 64
max_iters = 10100
eval_interval = 250
eval_iters = 250
n_embd = 576
n_layer = 10
n_head = 8
dropout = 0.3
warmup_iters = 5000

learning_rates = [3.5e-4, 1e-4, 5e-5, 1e-5, 7e-6, 3e-5, 5e-6]  # Added more learning rates
optimizers = ['Lamb', 'SGD', 'AdamW', 'RMSprop', 'Adagrad']

cornell_files = ['training_data/movie_lines.txt']
personality_files = 'training_data/personality.csv'

with open('chars.pkl', 'rb') as f:
    allowed_chars = pickle.load(f)

def clean_text_cornell(text):
    components = text.split('+++$+++')
    if len(components) >= 5:
        dialogue = components[4].strip()
        dialogue = ''.join([c for c in dialogue if c in allowed_chars])
        return dialogue

# Function to clean Facebook personality CSV
def clean_text_FB(text):
    text = re.sub(r'^\d+,\s*', '', text)  # Remove leading number and comma
    text = re.sub(r'\s+([?.!,])', r'\1', text)  # Fix spacing around punctuation
    text = re.sub(r'([?.!,])(\w)', r'\1 \2', text)  # Ensure space after punctuation
    text = re.sub(r'(^|[.!?]\s+)(\w)', lambda m: m.group(1) + m.group(2).upper(), text)  # Capitalize sentences
    text = re.sub(r'\bi\b', 'I', text)  # Capitalize standalone "i" to "I"
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra whitespace
    text = ''.join([c for c in text if c in allowed_chars])  # Filter out characters not in allowed_chars
    return text

# Function to pad chunk to block_size
def pad_chunk(encoded_chunk, block_size):
    if len(encoded_chunk) < block_size:
        padding = [0] * (block_size - len(encoded_chunk))
        encoded_chunk = torch.cat((encoded_chunk, torch.tensor(padding, dtype=torch.long)), dim=0)
    return encoded_chunk

# Function to clean Cornell movie corpus
def clean_cornell_movie_corpus(file_paths):
    cleaned_text = ""
    for file_path in file_paths:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
            for line in lines:
                cleaned_text += clean_text_cornell(line) + " "
    return cleaned_text

# Function to clean Facebook personality CSV
def clean_facebook_personality_csv(file_path):
    df = pd.read_csv(file_path)
    cleaned_data = ""
    for column in df.columns:
        if df[column].dtype == 'object':  # Process only string columns
            for entry in df[column]:
                cleaned_data += clean_text_FB(str(entry)) + " "
    return cleaned_data.strip()

# Load full_text using cleaned data
cornell_text = clean_cornell_movie_corpus(cornell_files)
personality_text = clean_facebook_personality_csv(personality_files)
full_text = cornell_text + personality_text

# Determine vocabulary size based on allowed_chars
vocab_size = len(allowed_chars)

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedFoward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        y = self.sa(x)
        x = self.ln1(x + y)
        y = self.ffwd(x)
        x = self.ln2(x + y)
        return x


class GPTLanguageModel(nn.Module):
    def __init__(self, vocab_size, freeze_blocks=False):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.apply(self._init_weights)

        if freeze_blocks:
            for param in self.blocks.parameters():
                param.requires_grad = False

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, index, targets=None):
        B, T = index.shape
        tok_emb = self.token_embedding_table(index.to(self.device))
        pos_emb = self.position_embedding_table(torch.arange(T, device=self.device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, index, max_new_tokens):
        for _ in range(max_new_tokens):
            index_cond = index[:, -block_size:]
            logits, loss = self.forward(index_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            index_next = torch.multinomial(probs, num_samples=1)
            index = torch.cat((index, index_next), dim=1)
        return index
    

def fine_tune_model(base_model_path, cornell_files, personality_files, learning_rate, optimizer_name, device):
    cornell_text = clean_cornell_movie_corpus(cornell_files)
    personality_text = clean_facebook_personality_csv(personality_files)
    full_text = cornell_text + personality_text

    with open(base_model_path, 'rb') as f:
        model = pickle.load(f)
    model.device = device
    model.to(device)

    if optimizer_name == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    elif optimizer_name == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    elif optimizer_name == 'RMSprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)
    elif optimizer_name == 'Adagrad':
        optimizer = torch.optim.Adagrad(model.parameters(), lr=learning_rate)
    elif optimizer_name == 'Lamb':
        optimizer = Lamb(model.parameters(), lr=learning_rate)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=learning_rate, total_steps=max_iters)
    criterion = nn.CrossEntropyLoss()

    for step in range(max_iters):
        if step % eval_interval == 0:
            print(f"Step {step}/{max_iters} with {optimizer_name} and learning rate {learning_rate}")

        start_pos = random.randint(0, len(full_text) - batch_size - 1)
        chunk = full_text[start_pos:start_pos + batch_size]
        encoded_chunk = torch.tensor([allowed_chars.index(c) for c in chunk], dtype=torch.long)

        # Pad the chunk to match block_size
        encoded_chunk = pad_chunk(encoded_chunk, block_size)

        assert len(encoded_chunk) == block_size, f"Chunk size mismatch: {len(encoded_chunk)} != {block_size}"

        input_data = encoded_chunk[:-1].unsqueeze(0).to(device)
        target_data = encoded_chunk[1:].unsqueeze(0).to(device)

        optimizer.zero_grad()
        logits, _ = model(input_data)
        loss = criterion(logits.squeeze(0), target_data.squeeze(0))
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        if step % eval_interval == 0:
            print(f"Step {step}: Loss = {loss.item()}")

    model_path = f'{base_model_path}-{optimizer_name}-{learning_rate}-{loss.item():.2f}.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"Fine-tuned model saved to {model_path}")

# Fine-tune the model with specified optimizers and learning rates
base_model_path = 'model-01.pkl'

for optimizer_name in optimizers:
    for learning_rate in learning_rates:
        print(f"Fine-tuning with optimizer: {optimizer_name}, learning rate: {learning_rate}")
        fine_tune_model(base_model_path, cornell_files, personality_files, learning_rate, optimizer_name, device)