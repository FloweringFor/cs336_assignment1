import os
import regex as re
from collections import Counter
import multiprocessing as mp
from typing import BinaryIO, List, Tuple, Dict


def train_bpe(
        input_path: str,
        vocab_size: int,
        special_tokens: List[str],
) -> Tuple[Dict[int, bytes], List[Tuple[bytes, bytes]]]:
    # 第一阶段：初始化词表
    vocab = initialize_vocab(special_tokens)

    # 第二阶段：并行预分词与计数 (保持原样，这部分逻辑是稳的)
    with open(input_path, 'rb') as f:
        num_chunks = mp.cpu_count()
        boundaries = find_chunk_boundaries(f, num_chunks, b'<|endoftext|>')
    task = [
        (input_path, boundaries[i], boundaries[i + 1], special_tokens)
        for i in range(len(boundaries) - 1)
    ]
    with mp.Pool(processes=num_chunks) as pool:
        chunk_counters = pool.map(pretokenize_worker, task)
    word_counts = Counter()
    for c in chunk_counters:
        word_counts.update(c)

    # 第三阶段：循环合并
    merge = []
    stat = Counter()
    pair_to_words = {}

    # 1. 初始建立索引
    for word, count in word_counts.items():
        for i in range(len(word) - 1):
            pair = (word[i], word[i + 1])
            stat[pair] += count
            if pair not in pair_to_words:
                pair_to_words[pair] = set()
            pair_to_words[pair].add(word)

    # 使用 current_vocab_size 确保 ID 分配严格连续且不越界
    while len(vocab) < vocab_size:
        # 过滤掉 count <= 0 的项，防止由于逻辑残留导致的错误合并
        actual_stats = {k: v for k, v in stat.items() if v > 0}
        if not actual_stats:
            break

        # A. 找出现频率最高且字节序最小的 pair (满足作业要求的 tie-breaking)
        best_pair = max(actual_stats.items(), key=lambda x: (x[1], x[0]))[0]

        # B. 严格同步更新 vocab 和 merge
        new_token_bytes = best_pair[0] + best_pair[1]
        vocab[len(vocab)] = new_token_bytes
        merge.append(best_pair)

        # C. 【核心修复】局部更新：确保受影响单词的所有 pair 引用都被刷新
        target_words = list(pair_to_words.get(best_pair, set()))

        # 必须先把这个 best_pair 的索引彻底清空，因为它即将被合并掉
        if best_pair in pair_to_words:
            del pair_to_words[best_pair]
        if best_pair in stat:
            del stat[best_pair]

        for old_word in target_words:
            if old_word not in word_counts:
                continue

            count = word_counts[old_word]

            # 1. 彻底清除旧词在所有相关 pair 中的计数和索引
            for i in range(len(old_word) - 1):
                p = (old_word[i], old_word[i + 1])
                if p in stat:
                    stat[p] -= count
                    if stat[p] <= 0:
                        del stat[p]
                if p in pair_to_words:
                    pair_to_words[p].discard(old_word)

            # 2. 生成新词 (合并逻辑)
            new_word = []
            i = 0
            while i < len(old_word):
                if i < len(old_word) - 1 and (old_word[i], old_word[i + 1]) == best_pair:
                    new_word.append(new_token_bytes)
                    i += 2
                else:
                    new_word.append(old_word[i])
                    i += 1
            new_word = tuple(new_word)

            # 3. 将新词重新加入 word_counts、stat 和索引
            # 注意：新词可能已经存在于 word_counts 中，所以要累加 count
            del word_counts[old_word]
            word_counts[new_word] = word_counts.get(new_word, 0) + count

            for i in range(len(new_word) - 1):
                p = (new_word[i], new_word[i + 1])
                stat[p] += count
                if p not in pair_to_words:
                    pair_to_words[p] = set()
                pair_to_words[p].add(new_word)

    return vocab, merge


def initialize_vocab(special_tokens: List[str]) -> Dict[int, bytes]:
    # 1. 基础 256 个字节 (ID: 0-255)
    # 记得使用 bytes([i]) 而不是 bytes(i)
    vocab = {i: bytes([i]) for i in range(256)}
    # 2. 加入特殊标记 (ID 从 256 开始往后排)
    for i, token in enumerate(special_tokens):
        vocab[i + 256] = token.encode('utf-8')
    return vocab


# 子进程要干的事
def pretokenize_worker(args):
    path, start, end, special_tokens = args
    counts = Counter()

    sorted_special = sorted(special_tokens, key=len, reverse=True)
    split_pattern = b"|".join([re.escape(t.encode('utf-8')) for t in sorted_special])
    PAT = rb"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    with open(path, 'rb') as f:
        f.seek(start)
        chunk_data = f.read(end - start)

        documents = re.split(split_pattern, chunk_data) if split_pattern else [chunk_data]
        for doc in documents:
            for match in re.finditer(PAT, doc):
                word_bytes = match.group()  # 返回的是匹配到的一个子串bytes
                char_tuple = tuple(bytes([b]) for b in word_bytes)
                counts[char_tuple] += 1

    return counts


def find_chunk_boundaries(
        file: BinaryIO,
        desired_num_chunks: int,
        split_special_token: bytes,
) -> List[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))
