import os
import regex as re
from typing import BinaryIO
from collections import Counter, defaultdict
from typing import Iterable, Iterator, Optional
import multiprocessing
import argparse
import json
import time
import psutil
import heapq

def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Find byte offsets in the file to split it into chunks without breaking documents.
    """
    assert isinstance(split_special_token, bytes), "Special token must be bytes"

    # Get total file size and reset pointer
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    print(f"File size: {file_size / (1024**2):.2f} MB")

    file.seek(0)
    
    

    chunk_size = file_size // desired_num_chunks
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Buffer size for searching the special token

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)
        while True:
            mini_chunk = file.read(mini_chunk_size)
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Locate the special token to define a safe boundary
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    return sorted(set(chunk_boundaries))

def process_chunk(
    file_path: str,
    start: int,
    end: int,
    pat_string: str,
    combined_escape: str
):
    """
    Worker function to count token frequencies in a specific file chunk.
    """
    local_counts = Counter()

    with open(file_path, "rb") as f:
        f.seek(start)
        # Read the assigned chunk and decode with error handling
        chunk_bytes = f.read(end - start)
        chunk_text = chunk_bytes.decode('utf-8', errors='ignore')

        # Split text into documents using escaped special tokens
        # docs example: ["Once upon a time", "Suddenly, it ended"]
        if combined_escape:
            docs = re.split(combined_escape, chunk_text)
        else: 
            docs = [chunk_text]
        
        # Pre-tokenize each document based on GPT-2 regex rules
        for doc in docs:
            for match in re.finditer(pat_string, doc):
                # Convert matched string to a byte tuple for BPE processing
                # example: b'hello' -> (104, 101, 108, 108, 111)
                token_tuple = tuple(match.group().encode('utf-8'))
                # example: Counter({(73, 39, 109): 1,...})
                local_counts[token_tuple] += 1
    return local_counts

def BPE_tokenizer(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str]
):
    """
    Main function to train BPE tokenizer and generate vocabulary and merges.
    """
    # 1. Initialize basic vocabulary (0-255 bytes)
    # example: {0: b'\x00', 97: b'a', 255: b'\xff'}
    vocab = {i: bytes([i]) for i in range(256)}

    # 2. Add special tokens to vocabulary with sequential IDs
    for i in special_tokens:
        vocab[len(vocab)] = i.encode('utf-8')

    # Set initial ID counter to the last assigned ID to ensure continuity
    current_vocab_size = len(vocab) - 1

    # 3. Parallel counting using Multiprocessing
    with open(input_path, "rb") as f:
        # Use the first special token as the document separator for chunking
        split_token_bytes = special_tokens[0].encode('utf-8')
        num_processes = 16
        boundaries = find_chunk_boundaries(f, 1000, split_token_bytes)

        # Build regex to protect special tokens from being split
        # example :['<\\|endoftext\\|>']
        escape_pattern = [re.escape(i) for i in special_tokens]
        combined_escape = "|".join(escape_pattern)
        
        # GPT-2 pre-tokenization regex pattern
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

        # Define tasks for each worker process
        tasks = []
        for i in range(len(boundaries) - 1):
            tasks.append((
                input_path,
                boundaries[i],
                boundaries[i+1],
                PAT,
                combined_escape
            ))

        # Distribute tasks across the process pool
        with multiprocessing.Pool(processes=num_processes) as pool:
            partial_results = pool.starmap(process_chunk, tasks)
        
        # Merge all partial counters into a single global counter
        # word_freqs = {(104, 105): 10, (104, 101): 5}
        word_freqs = Counter()
        for pr in partial_results:
            word_freqs.update(pr)


        # pair_stats: { (token_id1, token_id2): total_frequency }
        # Example: { (104, 105): 10 }
        pair_stats = defaultdict(int) 

        # pair_to_word_id: { (token_id1, token_id2): {word_index1, word_index2, ...} }
        # Example: { (104, 101): {0, 1} } -> pair (104, 101) exists in word 0 and word 1
        pair_to_word_id = defaultdict(set)

        # word_to_id: { (token_id1, token_id2, ...): unique_word_index }
        # Note: This is mainly used during initialization to map unique tuples to indices
        word_to_id = {}

        # id_to_word: { unique_word_index: [token_id1, token_id2, ...] }
        # Example: { 0: [104, 105] } -> word 0 is the sequence for "hi"
        id_to_word = {}

        word_id_counter = 0

        words_freq_list = []
    # Initializing data structures from the pre-tokenized counts
    for word_tuple, freq in word_freqs.items():
        # Assign a unique integer ID to each unique word sequence
        word_to_id[word_tuple] = word_id_counter
        # Store the word as a list to allow in-place modification during merging
        id_to_word[word_id_counter] = list(word_tuple)
        current_word_id = word_id_counter
        words_freq_list.append(freq)

        word_id_counter += 1
        
        # Iterate through the word to populate pair frequencies and inverted index
        for i in range(len(word_tuple) - 1):
            pair = word_tuple[i: i+2]
            
            # Accumulate the global frequency of this specific pair
            pair_stats[pair] += freq
            
            # Add this word's ID to the set of words containing this pair
            # This is the "Inverted Index" that allows O(1) lookup during merging
            pair_to_word_id[pair].add(current_word_id)

    


    merges = []

    # 4. Iterative BPE merging process
    while len(vocab) < vocab_size:
        if not pair_stats:
            print("No more pairs to merge. Stopping early.")
            break

        # Count frequencies of all adjacent pairs
        # Select the most frequent pair to merge
        max_pair = max(
            pair_stats.items(), 
            key=lambda x: (x[1], vocab[x[0][0]], vocab[x[0][1]])
        )[0]
        
        # Register new token in vocabulary
        current_vocab_size += 1
        vocab[current_vocab_size] = vocab[max_pair[0]] + vocab[max_pair[1]] # (e.g., b'he' + b'll' → b'hell')
        merges.append((vocab[max_pair[0]], vocab[max_pair[1]]))

        target_words_id = pair_to_word_id[max_pair]
        for word_id in list(target_words_id):
            word_sequence = id_to_word[word_id]
            freq = words_freq_list[word_id]
            

            for i in range(0, len(word_sequence) - 1): 
                p = (word_sequence[i], word_sequence[i+1])
                pair_stats[p] -= freq

                if pair_stats[p] == 0:
                    del pair_stats[p]

                pair_to_word_id[p].discard(word_id)

                if not pair_to_word_id[p]:
                    del pair_to_word_id[p]

            new_word_sequence = []
            i = 0

            while i < len(word_sequence):
                if i < len(word_sequence) - 1 and (word_sequence[i], word_sequence[i+1]) == max_pair:
                    new_word_sequence.append(current_vocab_size)
                    i += 2

                else:
                    new_word_sequence.append(word_sequence[i])
                    i += 1

            id_to_word[word_id] = new_word_sequence
            
            for i in range(len(new_word_sequence)-1):
                new_pair = (new_word_sequence[i], new_word_sequence[i+1])
                pair_stats[new_pair] += freq
                pair_to_word_id[new_pair].add(word_id)

        
        if len(vocab) % 100 == 0:
            print(f"Current vocab size: {len(vocab)} / {vocab_size}")

    proc = psutil.Process()
    total_mem = proc.memory_info().rss
    for child in proc.children(recursive=True):
        total_mem += child.memory_info().rss
    print(f"Memory peak: {total_mem / 1024**3:.2f} GB")

    return vocab, merges # vocab: {int: bytes}

def bytes_to_unicode():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1))+ list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))
    
    

# Entry point protection for Windows multiprocessing
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BPE Tokenizer Training")

    parser.add_argument("--input_path", type=str, required=True, help="Path to text")
    parser.add_argument("--vocab_size", type=int, default=10000, help="Target vocab size")
    parser.add_argument("--vocab_path", type=str, default="vocab.json",help="Output path for vocab")
    parser.add_argument("--merge_path", type=str, default="merges.txt", help="Output path for merges")
    #    parser.add_argument("--text_path", type = str,  )

    args = parser.parse_args()

    # register the starting time
    start_time = time.time()
    # Call the function using the argument parsed from the command line
    raw_vocab, raw_merges = BPE_tokenizer(
        input_path=args.input_path,
        vocab_size=args.vocab_size,
        special_tokens=["<|endoftext|>"]
    )
    # time spent on training
    duration_hours = (time.time() - start_time) / 3600
    print(f"Training took {duration_hours:.2f} hours")

    byte_encoder = bytes_to_unicode()

    vocab = {}

    vocab_parent_dir = os.path.dirname(args.vocab_path)
    if vocab_parent_dir:
        os.makedirs(vocab_parent_dir, exist_ok=True)

    for token_id, token_bytes in raw_vocab.items():
        token_str = "".join(byte_encoder[b] for b in token_bytes)
        # vocab example: {354: "Hello", 128: "Ā", ...}
        vocab[token_str] = token_id

    with open(args.vocab_path, "w", encoding='utf-8') as f:
        # ensure_ascii=False to ensure non-ascii unicode won't be transformed to \uXXXX
        json.dump(vocab, f, indent=4, ensure_ascii=False)

    merge_parent_dir = os.path.dirname(args.merge_path)
    if merge_parent_dir:
        os.makedirs(merge_parent_dir, exist_ok=True)

    with open(args.merge_path, "w", encoding='utf-8') as f:
        # add a version description
        f.write("#version: 0.1\n")
        for b1, b2 in raw_merges:
            s1 = "".join(byte_encoder[b] for b in b1)
            s2 = "".join(byte_encoder[b] for b in b2)
            f.write(f"{s1} {s2}\n")

    print(f"Successfully saved:")
    print(f"- Vocabulary: {os.path.abspath(args.vocab_path)}")
    print(f"- Merges: {os.path.abspath(args.merge_path)}")

class Node:
    def __init__(self, value):
        self.value: bytes = value 
        self.prev: Optional['Node'] = None 
        self.next: Optional['Node'] = None 
        self.remove: bool = False

class Tokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        # vocab: {int: bytes}
        self.vocab = vocab
        # merges:[(bytes, bytes),...]
        self.merges = merges
        # token_str_to_id: {str: id}
        # self.token_str_to_int = {v.decode('utf-8'): k for k, v in vocab.items()}
        self.bytes_to_int = {v: k for k, v in vocab.items()}
        self.special_tokens = special_tokens or []
        # bytes to unicode projection
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        self.token_str_to_int = {}
        for st in self.special_tokens:
            st_bytes = st.encode('utf-8')
            if st_bytes in self.bytes_to_int:
                self.token_str_to_int[st] = self.bytes_to_int[st_bytes]
        self.ranks = {merge: rank for rank, merge in enumerate(merges)}

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        byte_encoder = bytes_to_unicode() # {int: str}
        byte_decoder = {v: k for k, v in byte_encoder.items()} # {str: int}
        with open(vocab_filepath, "r", encoding='utf-8') as f:
            sorted_vocab = json.load(f) # {str[int]: str}
            vocab = {}

            for token_str, token_id in sorted_vocab.items(): 
                token_bytes = bytes([byte_decoder[char] for char in token_str])
                vocab[int(token_id)] = token_bytes #  {id: bytes} 

             
        with open(merges_filepath, "r", encoding='utf-8') as f:
            merges = []
            for line in f.read().splitlines()[1:]:
                s1, s2 = line.split()
                b1 = bytes([byte_decoder[c] for c in s1])
                b2 = bytes([byte_decoder[c] for c in s2])
                merges.append((b1, b2))
            return cls(vocab, merges, special_tokens)   

    def encode(
            self,
            text:str
    ) -> list[int]:
        count = 0
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        encoded_words = []
        if not self.special_tokens:
            docs = [text]
        else:
            sorted_special = sorted(self.special_tokens, key=len, reverse=True)
            combined_escape = "(" + "|".join(re.escape(i) for i in sorted_special) + ")"
            docs = [d for d in re.split(combined_escape, text) if d is not None and len(d) > 0]

        for doc in docs: 
            if doc is None or len(doc) == 0:  # 增加此行，过滤 None 和空字符串
                continue
            if doc in self.token_str_to_int:
                encoded_words.append(self.token_str_to_int[doc])
            else:
                for match in re.finditer(PAT, doc):
                    word_text = match.group()
                    word_bytes = word_text.encode('utf-8')
                    if len(word_bytes) == 0:
                        continue    
                    if len(word_bytes) < 2:
                        # Single character words don't need merging                 
                        encoded_words.append(self.bytes_to_int[word_bytes])
                        continue

                    # 1. Initialize Doubly Linked List
                    nodes = [Node(bytes([b])) for b in word_bytes]
                    for i in range(len(nodes)):
                        if i > 0: nodes[i].prev = nodes[i-1]
                        if i < len(nodes) - 1: nodes[i].next = nodes[i+1]

                    head = nodes[0]

                    # 2. Initialize Heap
                    queue = []
                    for i in range(len(nodes) - 1):
                        pair = (nodes[i].value, nodes[i+1].value)
                        if pair in self.ranks:
                            # Store as (rank, left_node, right_node)
                            # The heap will sort by rank automatically
                            count += 1
                            heapq.heappush(queue, (self.ranks[pair], count, nodes[i], nodes[i+1]))
                    while queue:
                        # pop out the smallest value
                        rank, _, left, right = heapq.heappop(queue)
                        if left.remove or right.remove or left.next is not right:
                            continue

                        if self.ranks.get((left.value, right.value)) != rank:
                            continue

                        left.value = left.value + right.value
                        right.remove = True

                        left.next = right.next
                        if right.next:
                            right.next.prev = left

                        if left.next:
                            new_pair_right = (left.value, left.next.value)
                            if new_pair_right in self.ranks:
                                count += 1
                                heapq.heappush(queue, (self.ranks[new_pair_right], count, left, left.next))

                        if left.prev:
                            new_pair_left = (left.prev.value, left.value)
                            if new_pair_left in self.ranks:
                                count += 1
                                heapq.heappush(queue, (self.ranks[new_pair_left], count, left.prev, left))

                    curr = head
                    while curr:
                        if not curr.remove:
                            if curr.value in self.bytes_to_int:
                                encoded_words.append(self.bytes_to_int[curr.value])
                            else:
                                raise KeyError(f"Token {curr.value} not found in vocab")
                        curr = curr.next

        return encoded_words
                

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
       
       for text_chunk in iterable:
            token_ids = self.encode(text_chunk)

            yield from token_ids

    def decode(self, ids: list[int]) -> str:
        
        all_bytes = b"".join([self.vocab[id] for id in ids])
        decoded_text = all_bytes.decode('utf-8', errors='replace')
        

        return decoded_text




