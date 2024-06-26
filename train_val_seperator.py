from multiprocessing import Pool
from transformers import GPT2Tokenizer
import gc
import os
import tqdm

def clean_text(text):
    return text

def process_file_in_chunks(file_path, chunk_size):
    """Generator to yield chunks of data from a file."""
    with open(file_path, 'r', encoding='utf-8') as file:
        while True:
            chunk = file.read(chunk_size)
            if not chunk:
                break
            yield chunk

def create_vocab_file(text, vocab_file_path):
    vocab = sorted(set(text))
    with open(vocab_file_path, 'w', encoding='utf-8') as f:
        for char in vocab:
            f.write(char)

def process_chunk(chunk, tokenizer, max_length=1024):
    # Clean and truncate chunk if necessary
    cleaned_chunk = clean_text(chunk[:max_length])
    
    # Tokenize the cleaned chunk
    tokenized_chunk = tokenizer.encode(cleaned_chunk, add_special_tokens=True)
    
    return tokenized_chunk

def process_and_write_chunk(args):
    """Process a chunk of data and write results to disk."""
    i, chunk, total_chunks = args  # Unpack the arguments

    # Determine file path based on chunk index
    if i < 0.8 * total_chunks:
        file_path = 'training_data/train_split.txt'
    else:
        file_path = 'training_data/val_split.txt'

    try:
        with open(file_path, 'a', encoding='utf-8') as file:
            file.write(chunk + '\n')  # Write the chunk
    except Exception as e:
        print(f"Error writing to {file_path}: {e}")

    del chunk
    gc.collect()

def main():
    chunk_size = 102400  # Adjust this value as needed based on performance and memory constraints

    input_files = [f"training_data/C4_200M.tsv-0000{i}-of-00010" for i in range(10)]
    
    for input_file in input_files:
        # Calculate the total size of the input file
        total_size = os.path.getsize(input_file) 

        # Calculate total_chunks dynamically
        total_chunks = total_size // chunk_size

        # Generator for chunks of data from the input file
        chunk_gen = process_file_in_chunks(input_file, chunk_size)

        with Pool(32) as pool:
            # Prepare arguments for process_and_write_chunk
            args = ((i, chunk, total_chunks) for i, chunk in enumerate(chunk_gen, start=1))

            # Use tqdm to show progress bar
            for _ in tqdm.tqdm(pool.imap_unordered(process_and_write_chunk, args), total=total_chunks, unit="chunks"):
                pass  # Progress bar updates handled within tqdm

if __name__ == '__main__':
    main()
