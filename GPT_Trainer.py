import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.nn.utils.prune as prune
import mmap
import random
import pickle
import re
import numpy as np
from pytorch_lamb import Lamb
from torch.cuda.amp import autocast, GradScaler

# Check if CUDA is available and if so, set the device accordingly
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# Define the parameters for the model and training
block_size = 128
batch_size = 64
max_iters = 11150
eval_interval = 500
eval_iters = 500
n_embd = 512
n_layer = 10
n_head = 8
dropout = 0.3
warmup_iters = 5000

# Define the learning rates and optimizers to test
learning_rates = [3.5e-4, 1e-4, 5e-5, 1e-5, 7e-6, 3e-5, 5e-6]  # Added more learning rates
optimizers = ['Lamb', 'SGD', 'AdamW', 'RMSprop', 'Adagrad']

# Read characters from the vocabulary file
with open("training_data/vocab.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Define characters to remove
unwanted_chars = ''.join([
    '\x00',  # null character
    '', '', '©',  # other unwanted symbols
    # Add any other characters you want to remove
])

# Filter out unwanted characters
cleaned_chars = sorted(list(set(text) - set(unwanted_chars)))

# Save the cleaned vocabulary
with open("training_data/cleaned_vocab.txt", "w", encoding="utf-8") as f:
    f.write(''.join(cleaned_chars))

vocab_size = len(cleaned_chars)
print(cleaned_chars)

with open('chars.pkl', 'wb') as f:
    pickle.dump(cleaned_chars, f)
string_to_int = {ch: i for i, ch in enumerate(cleaned_chars)}
int_to_string = {i: ch for i, ch in enumerate(cleaned_chars)}
encode = lambda s: [string_to_int.get(c, string_to_int[' ']) for c in s]  # default to space for unknown chars
decode = lambda l: ''.join([int_to_string[i] for i in l])

def clean_text(text):
    # Remove null values
    text = text.replace('\x00', '')
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # Remove emojis
    text = re.sub(r'[^\w\s,]', '', text, flags=re.UNICODE)
    
    # Remove text within brackets and the brackets
    text = re.sub(r'\[.*?\]', '', text)
    
    # Remove specific unwanted patterns
    text = re.sub(r'node r.js -o \S+', '', text)
    text = re.sub(r'SIZE: \d+ bytes SHA1: \w+ SHA256: \w+ SHA512: \w+', '', text)
    
    # Remove non-English characters
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Remove numbers without spaces
    text = re.sub(r'(?<!\s)\d+', '', text)

    
    return text

def get_random_chunk(split, block_size, batch_size):
    filename = "training_data/train_split.txt" if split == 'train' else "training_data/val_split.txt"
    with open(filename, 'rb') as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            file_size = len(mm)
            start_pos = random.randint(0, file_size - block_size * batch_size)
            mm.seek(start_pos)
            block = mm.read(block_size * batch_size - 1)
            decoded_block = block.decode('utf-8', errors='ignore').replace('\r', '')
            cleaned_block = clean_text(decoded_block)
            data = torch.tensor(encode(cleaned_block), dtype=torch.long)
    return data


def get_batch(split):
    data = get_random_chunk(split)
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)  # Move data to the correct device
    return x, y


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
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, index, targets=None):
        B, T = index.shape
        tok_emb = self.token_embedding_table(index)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
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

torch.cuda.empty_cache()

# Load the model and optimizer state
model = GPTLanguageModel(vocab_size)
print('Loading model parameters...')

try:
    with open('model-01.pkl', 'rb') as f:
        model = pickle.load(f)
    print('Loaded successfully!')
except FileNotFoundError:
    print('No pre-trained model found, starting from scratch.')

# Move the model to the appropriate device (GPU or CPU)
model = model.to(device)

# Apply model pruning
for name, module in model.named_modules():
    if isinstance(module, nn.Linear):
        prune.l1_unstructured(module, name='weight', amount=0.2)

#set_seed(37)  # Ensure reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# Define the loss estimation function
@torch.no_grad()
def estimate_loss():
    model.eval()
    out = {}
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

scaler = GradScaler()
#Initialize the optimizer to None
optimizer = None

# Outer loop for the learning rates
for lr in learning_rates:
    # Inner loop for the optimizers
    for opt in optimizers:
        # Set the optimizer based on the current optimizer in the loop
        if opt == 'Lamb':
            optimizer = Lamb(model.parameters(), lr=lr)
        elif opt == 'SGD':
            optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        elif opt == 'AdamW':
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        elif opt == 'RMSprop':
            optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)
        elif opt == 'Adagrad':
            optimizer = torch.optim.Adagrad(model.parameters(), lr=lr)

        print(f'Current optimizer: {opt}, learning rate: {lr}')

        for step in range(max_iters):
            X, Y = get_batch('train')

            # Perform the forward pass and calculate loss under autocast
            with autocast():
                logits, loss = model(X, Y)

            # Scale the loss for mixed-precision training
            scaler.scale(loss).backward()

            # Unscales the gradients and calls optimizer.step()
            scaler.step(optimizer)

            # Updates the scale for next iteration
            scaler.update()

            if step % eval_interval == 0:
                losses = estimate_loss()
                print(f"Step {step}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

print("Training completed.")

# Save the model
with open('model-01.pkl', 'wb') as f:
    pickle.dump(model, f)
print("Model saved.")

