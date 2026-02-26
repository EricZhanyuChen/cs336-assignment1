import numpy as np
from cs336_basics.tokenizer import Tokenizer
import argparse
import os
import time

def parse_args():
    parser = argparse.ArgumentParser(description="Encode texts into npy files with progress tracking")
    parser.add_argument("--file", required=True, choices=["owt", "tinystories"], help="Dataset name")
    return parser.parse_args()

def process_text(name):
    if name == "owt":
        vocab_path = "data/owt_32k.json"
        merge_path = "data/owt_32k.txt"
        splits = {"train": "data/owt_train.txt", "val": "data/owt_valid.txt"}
    else:
        vocab_path = "data/ts_vocab.json"
        merge_path = "data/ts_merge.txt"
        splits = {"train": "data/TinyStoriesV2-GPT4-train.txt", "val": "data/TinyStoriesV2-GPT4-valid.txt"}

    tokenizer = Tokenizer.from_files(
        vocab_path,
        merge_path,
        special_tokens=["<|endoftext|>"]
    )

    for split_name, input_path in splits.items():
        if not os.path.exists(input_path):
            print(f"Warning: File not found {input_path}")
            continue

        print(f"Starting encoding for {name} {split_name}...")
        all_ids = []
        start_time = time.time()
        
        with open(input_path, "r", encoding="utf-8") as f:
            # Processing in chunks to provide visual progress
            # instead of using list() which blocks until the end
            chunk_size = 10000 
            count = 0
            
            while True:
                # Read a chunk of lines
                lines = [f.readline() for _ in range(chunk_size)]
                lines = [l for l in lines if l] # Remove empty lines
                if not lines:
                    break
                
                # Encode the current chunk
                chunk_ids = list(tokenizer.encode_iterable(lines))
                all_ids.extend(chunk_ids)
                
                count += len(lines)
                elapsed = time.time() - start_time
                print(f"Processed {count} lines... ({len(all_ids):,}/tokens) Time: {elapsed:.2f}s", end="\r")

        output_path = f"data/{name}_{split_name}.npy"
        # Using uint16 as required to save space 
        np.save(output_path, np.array(all_ids, dtype=np.uint16))
        print(f"\nSuccessfully saved to {output_path}. Total tokens: {len(all_ids):,}")

if __name__ == "__main__":
    args = parse_args()
    process_text(args.file)