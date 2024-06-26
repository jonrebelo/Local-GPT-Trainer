import re
import os
import gc
from multiprocessing import Pool
import tqdm
import random


allowed_chars = set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789?!.,:;[] \n\t')

def clean_text(text):
    print(f"Original Text:\n{text[:1000]}...\n")  # Print first 500 characters as a snippet
    if text is None:
        print("Warning: Received 'None' text input.")
        return ""
    
    #Remove null characters
    text = text.replace('\x00', '')

    #Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\S+\.\S+', '', text)
    
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'\{\{[^}]*\}\}', '', text)
    
    #Remove dates and timestamps
    text = re.sub(r'\b\d{8}T\d{2}:\d{2}:\d{2}Z\b', '', text)
    
    # Remove non-alphanumeric characters except for specified punctuation marks
    text = re.sub(r'[^\w\s?!.,:;]', '', text, flags=re.UNICODE)

    # Remove repeating words
    text = re.sub(r'\b(\w+)\s+\1\b', r'\1', text)
    
    #Remove words with "_"
    text = re.sub(r'\b\w+_\w+\b', '', text)

    # Collapse multiple spaces into a single space
    text = ' '.join(text.split())
    
    # Ensure only allowed characters are retained
    text = ''.join(filter(lambda x: x in allowed_chars, text))

    # Remove leading and trailing whitespace
    text = text.strip()
    #print(f"\nCleaned Text:\n{text[:500]}...\n")
    #print("=" * 50)

    return text

def create_vocab_file(text, vocab_file_path):
    vocab = sorted(set(text))
    with open(vocab_file_path, 'w', encoding='utf-8') as f:
        for char in vocab:
            f.write(char)
            

def process_file_in_chunks(file_path, chunk_size):
    with open(file_path, 'r', encoding='utf-8') as file:
        while True:
            chunk = file.read(chunk_size)
            if not chunk:
                break
            yield chunk

def process_chunk(chunk):
    cleaned_chunk = clean_text(chunk)
    return cleaned_chunk

def process_and_write_chunk(args):
    i, chunk, total_chunks = args  # Unpack the arguments

    cleaned_chunk = clean_text(chunk)

    # Determine file path based on chunk index
    if i < 0.8 * total_chunks:
        file_path = 'training_data/train_split.txt'
    else:
        file_path = 'training_data/val_split.txt'

    try:
        with open(file_path, 'a', encoding='utf-8') as file:
            file.write(cleaned_chunk + '\n')  # Write chunk followed by a newline
    except Exception as e:
        print(f"Error writing to {file_path}: {e}")

    #print(f"Processed chunk {i} with size {len(cleaned_chunk.encode('utf-8'))} bytes, wrote to {file_path}")

    del cleaned_chunk
    gc.collect()

def print_random_snippets(data_path, num_snippets=3):
    with open(data_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    selected_indices = random.sample(range(len(lines)), num_snippets)
    for idx in selected_indices:
        snippet = lines[idx].strip()
        cleaned_text = clean_text(snippet)

        print("=" * 50)
        print(f"Random Original Snippet:")
        print(snippet[:500])  # Print first 500 characters as a snippet
        print("-" * 50)
        print(f"Random Cleaned Snippet:")
        print(cleaned_text[:500])  # Print first 500 characters as a snippet
        print("=" * 50)


def main():
    # Specify paths to your training and validation data
    train_data_path = 'training_data/train_split.txt'
    val_data_path = 'training_data/val_split.txt'

    chunk_size = 10240  # Adjust this value as needed based on performance and memory constraints

    # Calculate the total size of the input file
    total_size = os.path.getsize("training_data/data.txt")

    # Calculate total_chunks dynamically
    total_chunks = total_size // chunk_size

    # Generator for chunks of data from the XML file
    chunk_gen = process_file_in_chunks("training_data/data.txt", chunk_size)

    with Pool(32) as pool:
        # Prepare arguments for process_and_write_chunk
        args = ((i, chunk, total_chunks) for i, chunk in enumerate(chunk_gen, start=1))

        # Use tqdm to show progress bar
        for _ in tqdm.tqdm(pool.imap_unordered(process_and_write_chunk, args), total=total_chunks, unit="chunks"):
            pass

    #Print random snippets from the training data after processing
    #print("Printing random snippets from training data:")
    #print_random_snippets(train_data_path)

if __name__ == '__main__':
    main()