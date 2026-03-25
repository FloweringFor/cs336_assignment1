from collections.abc import Iterable, Iterator
import json
from typing import Optional, List, Dict, Tuple

import regex as re
import cs336_basics.utils as utils


byte_encoder = utils.bytes_to_unicode()
byte_decoder = {v: k for k, v in byte_encoder.items()}


class Tokenizer:
    def __init__(
            self,
            vocab: Dict[int, bytes],
            merges: List[Tuple[bytes, bytes]],
            special_tokens: Optional[List[str]] = None
    ):
        """
        Construct a tokenizer from a given vocabulary, list of merges, and (optionally) a list of special tokens.
        """
        # 基础映射 id -> byte
        self.vocab = vocab
        # 反向映射 byte -> id
        self.byte_to_id = {b: i for i, b in vocab.items()}
        self.merges = {pair: rank for rank, pair in enumerate(merges)}
        # 特殊标记处理
        self.special_tokens = special_tokens or []

    @classmethod
    def from_files(
            cls,
            vocab_filepath: str,
            merges_filepath: str,
            special_tokens: Optional[List[str]] = None
    ):
        """
        Class method that constructs and return a Tokenizer from a serialized vocabulary and list of merges
        (in the same format that your BPE training code output) and (optionally) a list of special tokens
        """
        # 1. 加载 JSON
        with open(vocab_filepath, 'r', encoding='utf-8') as f:
            vocab_json = json.load(f)

        # 2. 核心还原：将映射字符串还原为 bytes
        vocab = {}
        for safe_str, idx in vocab_json.items():
            # 根据映射表变回真正的字节，比如 "Ġ" 变回 b' '
            original_bytes = bytes([byte_decoder[c] for c in safe_str])
            vocab[idx] = original_bytes

        # 3. 处理 Merges
        merges = []
        with open(merges_filepath, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split(' ')
                if len(parts) != 2: continue
                # 同样还原字节对
                p1 = bytes([byte_decoder[c] for c in parts[0]])
                p2 = bytes([byte_decoder[c] for c in parts[1]])
                merges.append((p1, p2))

        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> List[int]:
        """
        Encode an input text into a sequence of token IDs
        """
        # 核心：按长度从长到短排序，防止短标记“偷走”长标记的前缀
        sorted_special = sorted(self.special_tokens, key=len, reverse=True)
        special_pattern = "|".join([re.escape(t) for t in sorted_special])
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        if special_pattern:
            # 使用括号保留分隔符，确保返回的是 str 列表
            parts = re.split(f"({special_pattern})", text)
            # 过滤掉 split 产生的空字符串（这在正则匹配到开头/结尾时经常发生）
            parts = [p for p in parts if p]
        else:
            parts = [text]

        ids = []
        for part in parts:
            if part in self.special_tokens:
                ids.append(self.byte_to_id[part.encode('utf-8')])
            else:
                for match in re.finditer(PAT, part):
                    word_bytes = match.group().encode('utf-8')
                    word = tuple(bytes([b]) for b in word_bytes)

                    while len(word) >= 2:
                        pairs = [(word[i], word[i + 1]) for i in range(len(word) - 1)]
                        best_pair = min(pairs, key=lambda p: self.merges.get(p, float('inf')))

                        if best_pair not in self.merges:
                            break  # 没有可以合并的对了
                        new_word = []
                        i = 0
                        while i < len(word):
                            if i < len(word) - 1 and (word[i], word[i + 1]) == best_pair:
                                new_word.append(word[i] + word[i + 1])
                                i += 2
                            else:
                                new_word.append(word[i])
                                i += 1
                        word = tuple(new_word)
                    for token_bytes in word:
                        ids.append(self.byte_to_id[token_bytes])
        return ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        Given an iterable of strings (e.g., a Python file handle), return a generator that lazily yields token IDs.
        This is required for memory-efficient tokenization of large files that we cannot directly load into memory.
        """
        for line in iterable:
            token_ids = self.encode(line)
            yield from token_ids

    def decode(self, ids: List[int]) -> str:
        """
        Decode a sequence of token IDs into text
        """
        # 1. 直接拼接原始字节流
        # self.id_to_byte 是 byte_to_id 的反向映射
        full_bytes = b"".join([self.vocab[i] for i in ids])

        # 2. 整体解码一次
        # 如果此时还有非法字节（极罕见），errors='replace' 才作为最后的防线
        return full_bytes.decode('utf-8', errors='replace')
