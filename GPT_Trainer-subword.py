import os
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.nn.utils.prune as prune
import mmap
import random
import numpy as np
from pytorch_lamb import Lamb
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from transformers import GPT2TokenizerFast 
from torch.optim import AdamW


# Check if CUDA is available and if so, set the device accordingly
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
torch.cuda.empty_cache()

# Define the parameters for the model and training
block_size = 128
batch_size = 24
max_iters = 20100
eval_interval = 500
eval_iters = 500
n_embd = 640
n_layer = 14
n_head = 14
dropout = 0.20

# Define the learning rates and optimizers to test
learning_rates = [1e-4, 5e-5, 2e-5]
optimizer_dict = {
    'AdamW': AdamW,
    'Lamb': Lamb,
}

# Function to encode text using subword tokenizer
def encode_text(text, tokenizer):
    encoded = tokenizer.encode(text, return_tensors='pt').squeeze(0)
    return encoded

# Load existing tokenizer
tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
vocab_size = tokenizer.vocab_size



# Function to get a random chunk of encoded data from train_split.txt or val_split.txt
def get_random_chunk(split):
    filename = "training_data/train_split.txt" if split == 'train' else "training_data/val_split.txt"
    with open(filename, 'rb') as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            file_size = len(mm)
            start_pos = random.randint(0, file_size - block_size * batch_size)
            mm.seek(start_pos)
            block = mm.read(block_size * batch_size - 1)
            decoded_block = block.decode('utf-8', errors='ignore').replace('\r', '')

            # Tokenize and encode the text
            encoded_block = tokenizer.encode(decoded_block[:1024], return_tensors='pt').squeeze(0)  # Truncate to maximum length

            if encoded_block.size(0) > block_size:
                return encoded_block
            else:
                print("Encoded block is too small, retrying...")
                return get_random_chunk(split)

# Function to get a batch of data
def get_batch(split):
    data = get_random_chunk(split)

    if len(data) <= block_size:
        raise ValueError("Data length is less than or equal to block_size, cannot generate batch.")

    ix = torch.randint(0, len(data) - block_size, (batch_size,))
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
            # Print shape of index for debugging
            print(f"Index shape: {index.shape}")  # Check the shape

            # Ensure index has batch dimension if it's missing
            if index.dim() == 1:
                index = index.unsqueeze(0)  # Add batch dimension if missing
                print(f"After unsqueezing, Index shape: {index.shape}")  # Check the shape after unsqueezing

            # Ensure index has the correct shape for slicing
            if index.dim() != 2:
                raise ValueError(f"Unexpected index shape: {index.shape}. Expected 2-dimensional tensor.")

            # Slice the index to get the last block_size tokens
            index_cond = index[:, -block_size:]

            # Forward pass to generate the next token
            logits, loss = self.forward(index_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            index_next = torch.multinomial(probs, num_samples=1)

            # Concatenate the generated token to index for the next iteration
            index = torch.cat((index, index_next), dim=1)

        return index

# Function to set seed for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def freeze_layers(model, layers_to_freeze):
    for name, param in model.named_parameters():
        for layer_name in layers_to_freeze:
            if layer_name in name:
                param.requires_grad = False
                print(f"Freezing parameter: {name}")


def prune_layers(model, layers_to_prune, amount):
    for name, module in model.named_modules():
        for layer_name in layers_to_prune:
            if layer_name in name:
                for name, param in module.named_parameters():
                    prune.l1_unstructured(param, amount=amount)
                    print(f"Pruning parameter in module {name} with amount: {amount}")


# Specify layers to freeze and prune
layers_to_freeze = ['blocks.0', 'blocks.1', 'blocks.2', 'blocks.3']
layers_to_prune = ['blocks.4', 'blocks.5', 'blocks.6', 'blocks.7']
prune_amount = 0.2

# Function to train the model

def train_model(epochs, learning_rates=learning_rates, optimizer_dict=optimizer_dict, checkpoint_path='model_checkpoint_epoch_27.pt', layers_to_freeze=[], layers_to_prune=[], prune_amount=0.2):
    set_seed(37)  # Set seed for reproducibility
    best_val_loss = float('inf')
    best_config = None

    for lr in learning_rates:
        for optimizer_name in optimizer_dict:
            # Initialize model
            model = GPTLanguageModel(vocab_size).to(device)
            #freeze_layers(model, layers_to_freeze)
            #prune_layers(model, layers_to_prune, prune_amount)
            #model.apply(prune.remove)

            optimizer_class = optimizer_dict[optimizer_name]
            optimizer = optimizer_class(model.parameters(), lr=lr, weight_decay=0.005)
            scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, min_lr=1e-6)

            # Load checkpoint if it exists
            start_epoch = 1
            if os.path.exists(checkpoint_path):
                start_epoch, best_val_loss = load_checkpoint(model, optimizer, checkpoint_path)
            else:
                print("No Checkpoint Found")

            print(f"Training with optimizer: {optimizer_name}, learning rate: {lr}")

            scaler = GradScaler()
            early_stopping_patience = 5  # Adjusted early stopping patience

            for epoch in range(start_epoch, epochs + 1):
                model.train()
                total_loss = 0.0

                for iteration in range(1, max_iters + 1):
                    optimizer.zero_grad()  # Clear gradients
                    inputs, targets = get_batch('train')  # Get a batch of data

                    with autocast():
                        logits, loss = model(inputs, targets)  # Forward pass

                    scaler.scale(loss).backward()  # Backward pass with scaling
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
                    scaler.step(optimizer)  # Optimizer step
                    scaler.update()  # Update scaler
                    total_loss += loss.item()

                    # Evaluate the model at specified intervals
                    if iteration % eval_interval == 0:
                        val_loss = evaluate_model(model)
                        print(f"Epoch [{epoch}/{epochs}], Iteration [{iteration}/{max_iters}], "
                              f"Train Loss: {total_loss / eval_interval:.4f}, Val Loss: {val_loss:.4f}")
                        total_loss = 0.0

                        # Check if validation loss has improved
                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            best_config = (optimizer_name, lr)
                            print(f"Found new best configuration - Optimizer: {optimizer_name}, Learning Rate: {lr}, "
                                  f"Validation Loss: {val_loss:.4f}")
                            plateau_count = 0  # Reset plateau count
                        else:
                            plateau_count += 1

                        # Early stopping condition
                        if plateau_count >= early_stopping_patience:
                            print(f"Stopping early at epoch {epoch} due to no improvement in validation loss.")
                            break  # Exit inner loop, go to next optimizer or LR

                # Generate test prompts after each epoch
                print(f"\nGenerating test prompts after epoch {epoch}:")
                for _ in range(3):  # Generate 3 test prompts
                    starting_prompt = "The protagonist"
                    starting_token = encode_text(starting_prompt, tokenizer).to(device)
                    generated_sequence = model.generate(starting_token, max_new_tokens=100)
                    generated_text = tokenizer.decode(generated_sequence[0].tolist(), skip_special_tokens=True)
                    print(f"Prompt: '{starting_prompt}'\nGenerated Text: '{generated_text}'\n")

                # Adjust learning rate based on validation loss
                scheduler.step(val_loss)
                current_lr = optimizer.param_groups[0]['lr']
                print(f"Current learning rate: {current_lr:.8f}")

                # Save model checkpoint after each epoch
                save_checkpoint(model, optimizer, epoch, val_loss, checkpoint_path)

                # Check for early stopping
                if plateau_count >= early_stopping_patience:
                    break  # Exit outer loop, stop training early

            if plateau_count < early_stopping_patience:
                print(f"Completed training for optimizer: {optimizer_name}, learning rate: {lr}, "
                      f"Best validation loss: {best_val_loss:.4f}\n")

    print(f"Best configuration found - Optimizer: {best_config[0]}, Learning Rate: {best_config[1]}, "
          f"Validation Loss: {best_val_loss:.4f}")


# Function to evaluate the model on validation set
def evaluate_model(model):
    model.eval()  # Set the model to evaluation mode
    total_loss = 0.0

    with torch.no_grad():
        for _ in range(eval_iters):
            inputs, targets = get_batch('val')  # Get a batch of validation data
            logits, loss = model(inputs, targets)  # Forward pass
            total_loss += loss.item()

    return total_loss / eval_iters

# Function to save model checkpoint
def save_checkpoint(model, optimizer, epoch, val_loss, checkpoint_path):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
    }

    torch.save(checkpoint, f'model_checkpoint_epoch_{epoch}.pt')
    print(f"Checkpoint saved for epoch {epoch} with validation loss {val_loss:.4f}")

def load_checkpoint(model, optimizer, checkpoint_path='checkpoint.pth'):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        val_loss = checkpoint['val_loss']
        print(f"Checkpoint loaded. Resuming from epoch {start_epoch} with validation loss {val_loss:.4f}")
        return start_epoch, val_loss

    
# Example usage to train the model
train_model(epochs=50)
