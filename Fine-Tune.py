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

# Check if CUDA is available and set the device accordingly

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.empty_cache()


block_size = 128
batch_size = 64
max_iters = 40000
eval_interval = 1000
eval_iters = 1000
n_embd = 512
n_layer = 10
n_head = 8
dropout = 0.3
warmup_iters = 5000

learning_rates = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
optimizers = ['AdamW', 'SGD', 'RMSpop', 'Adagrad', 'Lamb']



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
    def __init__(self, vocab_size, device=None, freeze_blocks=False):
        super().__init__()
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
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



cornell_files = [
    'F:/00_Github/Local_LLM/training_data/movie_characters_metadata.txt',
    'F:/00_Github/Local_LLM/training_data/movie_conversations.txt',
    'F:/00_Github/Local_LLM/training_data/movie_lines.txt',
    'F:/00_Github/Local_LLM/training_data/movie_titles_metadata.txt'
]

def clean_text(text):
    # Customized cleaning function
    text = text.lower()  # Convert to lowercase
    text = text.replace('\n', ' ')  # Remove newline characters
    text = ''.join([c for c in text if c.isalnum() or c == ' '])  # Keep alphanumeric characters and spaces
    return text


def clean_cornell_movie_corpus(file_paths):
    cleaned_text = ""
    for file_path in file_paths:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()
            cleaned_text += clean_text(text)
    return cleaned_text

cornell_text = clean_cornell_movie_corpus(cornell_files)

print("Cleaning Facebook Personality CSV...")
personality_file = 'F:/00_Github/Local_LLM/training_data/personality.csv'

def clean_facebook_personality_csv(file_path):
    df = pd.read_csv(file_path)
    cleaned_data = ""
    for column in df.columns:
        if df[column].dtype == 'object':  # Process only string columns
            for entry in df[column]:
                cleaned_data += clean_text(str(entry))
    return cleaned_data

personality_text = clean_facebook_personality_csv(personality_file)

full_text = cornell_text + personality_text
chars = sorted(list(set(full_text)))
vocab_size = len(chars)  # Determine vocabulary size based on unique characters

def fine_tune_model(base_model_path, cornell_files, personality_file, learning_rate, optimizer_name, device):
    # Clean data
    print("Cleaning Cornell Movie Corpus...")
    cornell_text = clean_cornell_movie_corpus(cornell_files)
    print("Cleaning Facebook Personality CSV...")
    personality_text = clean_facebook_personality_csv(personality_file)

    # Concatenate and encode cleaned text
    full_text = cornell_text + personality_text
    chars = sorted(list(set(full_text)))
    vocab_size = len(chars)  # Determine vocabulary size based on unique characters

    # Load pre-trained model and move to appropriate device
    with open(base_model_path, 'rb') as f:
        model = pickle.load(f)
    model.device = device  # Manually set the device attribute
    model.to(device)

    # Define optimizer and criterion
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

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.1)

    criterion = nn.CrossEntropyLoss()

    # Fine-tuning loop
    for step in range(max_iters):
        if step % eval_interval == 0:
            print(f"Step {step}/{max_iters}")

        # Sample a random chunk of data
        start_pos = random.randint(0, len(full_text) - batch_size - 1)
        chunk = full_text[start_pos:start_pos + batch_size]

        # Encode the chunk
        encoded_chunk = torch.tensor([chars.index(c) for c in chunk], dtype=torch.long)

        # Prepare input and target tensors
        input_data = encoded_chunk[:-1].unsqueeze(0)
        target_data = encoded_chunk[1:].unsqueeze(0)

        # Move tensors to device
        input_data = input_data.to(device)
        target_data = target_data.to(device)

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        logits, _ = model(input_data)

        # Compute loss
        loss = criterion(logits.squeeze(0), target_data.squeeze(0))

        # Backward pass
        loss.backward()

        # Update parameters
        optimizer.step()

        # Step the learning rate scheduler
        scheduler.step()

        # Print loss periodically
        if step % eval_interval == 0:
            print(f"Step {step}: Loss = {loss.item()}")

    # Save the fine-tuned model
    model_path = f'{base_model_path}-{optimizer_name}-{learning_rate}-{loss.item():.2f}.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"Fine-tuned model saved to {model_path}")

# Example usage
base_model_path = 'model-01.pkl'
cornell_files = [
    'F:/00_Github/Local_LLM/training_data/movie_characters_metadata.txt',
    'F:/00_Github/Local_LLM/training_data/movie_conversations.txt',
    'F:/00_Github/Local_LLM/training_data/movie_lines.txt',
    'F:/00_Github/Local_LLM/training_data/movie_titles_metadata.txt'
]
personality_file = 'F:/00_Github/Local_LLM/training_data/personality.csv'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.empty_cache()

for optimizer_name in optimizers:
    for learning_rate in learning_rates:
        print(f"Fine-tuning with optimizer: {optimizer_name}, learning rate: {learning_rate}")

        # Call fine_tune_model with current optimizer and learning rate
        fine_tune_model(base_model_path, cornell_files, personality_file, learning_rate, optimizer_name, device)
