import torch
import torch.nn as nn
from torch.nn import functional as F
import re
import os
import gc
from multiprocessing import Pool, cpu_count
import tqdm

allowed_chars = set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789?!.,:; \n\t')

def clean_text(text):
    text = text.replace('\x00', '')
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^\w\s,]', '', text, flags=re.UNICODE)
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'node r.js -o \S+', '', text)
    text = re.sub(r'SIZE: \d+ bytes SHA1: \w+ SHA256: \w+ SHA512: \w+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'(?<!\s)\d+', '', text)
    text = text.strip()
    text = ' '.join(text.split())
    text = re.sub(r'==.*?==', '', text)
    text = re.sub(r'\[\[.*?\]\]', '', text)
    text = re.sub(r'{{.*?}}', '', text)
    return text

def create_vocab_file(text, vocab_file_path):
    vocab = sorted(set(text))
    with open(vocab_file_path, 'w', encoding='utf-8') as f:
        for char in vocab:
            f.write(char)

def process_file_in_chunks(file_path, chunk_size): 
    with open(file_path, "r", encoding="utf-8") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            yield chunk

def encode(s, string_to_int):
    return [string_to_int.get(c, string_to_int[' ']) for c in s]

def decode(l, int_to_string):
    return ''.join([int_to_string[i] for i in l])

def process_chunk(chunk):
    cleaned_chunk = clean_text(chunk)
    return cleaned_chunk

def process_chunk(chunk):
    cleaned_chunk = clean_text(chunk)
    return cleaned_chunk

def process_and_write_chunk(args):
    i, chunk, total_chunks, chunk_size = args
    cleaned_chunk = process_chunk(chunk)

    if i < 0.8 * total_chunks:
        with open('training_data/train_split.txt', 'a', encoding='utf-8') as train_file:
            train_file.write(cleaned_chunk + '\n')  # Write chunk followed by a newline
    else:
        with open('training_data/val_split.txt', 'a', encoding='utf-8') as val_file:
            val_file.write(cleaned_chunk + '\n')  # Write chunk followed by a newline

    del cleaned_chunk
    gc.collect()


def main():
    chunk_size = 12 * 2048 * 2048  # Adjust this value to experiment with different chunk sizes
    total_size = os.path.getsize("training_data/wiki-data.xml")
    total_chunks = total_size // chunk_size

    chunk_gen = process_file_in_chunks("training_data/wiki-data.xml", chunk_size)
    
    with Pool(12) as pool:  # Use 8 threads
        # Map with a generator to avoid loading all chunks into memory
        for _ in tqdm.tqdm(pool.imap_unordered(process_and_write_chunk, 
                                               ((i, chunk, total_chunks, chunk_size) for i, chunk in enumerate(chunk_gen))), 
                           total=total_chunks):
            pass

if __name__ == '__main__':
    main()
