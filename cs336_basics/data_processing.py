import os
import json
import numpy as np
import multiprocessing as mp
from tqdm import tqdm
import cs336_basics.tokenizer as tokenizer
import cs336_basics.bpe as bpe
import cs336_basics.utils as utils

# --- 1. 把 worker 移到最外层 ---
# 我们需要一个全局变量来存放每个进程自己的 tokenizer
_process_tokenizer = None
_process_eot_id = None


def init_worker(tokenizer_params):
    """每个子进程启动时都会运行一次这个函数"""
    global _process_tokenizer, _process_eot_id
    from cs336_basics.tokenizer import Tokenizer
    # 重新从参数加载 tokenizer
    v_path, m_path, s_tokens = tokenizer_params
    _process_tokenizer = Tokenizer.from_files(v_path, m_path, s_tokens)
    _process_eot_id = _process_tokenizer.byte_to_id[b'<|endoftext|>']


def worker_func(line):
    """这个函数现在在顶层，可以被 pickle"""
    # 处理逻辑：encode + 加上结束符 ID
    ids = _process_tokenizer.encode(line)
    return ids + [_process_eot_id]


# --- 2. 修改 process_split ---
def process_split(split_name, input_path, output_path, vocab_path, merges_path, special_tokens, num_proc):
    print(f"正在处理 {split_name} 数据...")

    # 准备传给子进程的初始化参数
    init_params = (vocab_path, merges_path, special_tokens)

    def chunk_generator(file_path, size=10000):
        with open(file_path, 'r', encoding='utf-8') as f:
            batch = []
            for line in f:
                line = line.strip()
                if line:
                    batch.append(line)
                if len(batch) >= size:
                    yield batch
                    batch = []
            if batch:
                yield batch

    all_ids = []

    # 关键：使用 initializer 将 tokenizer 分发给 112 个核
    with mp.Pool(num_proc, initializer=init_worker, initargs=(init_params,)) as pool:
        for batch in chunk_generator(input_path):
            # 现在的 worker_func 是顶层函数，不再报错
            results = list(pool.imap(worker_func, batch, chunksize=100))
            for res in results:
                all_ids.extend(res)

    ids_array = np.array(all_ids, dtype=np.uint16)
    ids_array.tofile(output_path)
    print(f"{split_name} 完成！Token 总数: {len(ids_array)}")


if __name__ == '__main__':
    byte_encoder = utils.bytes_to_unicode()

    path = "../datasets/TinyStoriesV2-GPT4-"
    special_tokens = ["<|endoftext|>"]

    vocab_size = 20000

    vocab, merges = bpe.train_bpe(f'{path}valid.txt', vocab_size, special_tokens)

    # vocab {int: bytes}
    # 注意：这里不再使用 errors='replace'，因为映射表保证了安全
    vocab_to_json = {}
    for idx, token_bytes in vocab.items():
        # 将 b"don'" 转换成类似 "don'" 的映射字符串
        safe_str = "".join(byte_encoder[b] for b in token_bytes)
        vocab_to_json[safe_str] = idx

    # 写入 JSON
    with open(f'{path}vocab.json', 'w', encoding='utf-8') as f:
        json.dump(vocab_to_json, f, indent=4, ensure_ascii=False)

    # 写入 Merges
    with open(f'{path}merges.txt', 'w', encoding='utf-8') as f:
        for p1, p2 in merges:
            s1 = "".join(byte_encoder[b] for b in p1)
            s2 = "".join(byte_encoder[b] for b in p2)
            f.write(f"{s1} {s2}\n")

    tok = tokenizer.Tokenizer.from_files(f'{path}vocab.json', f'{path}merges.txt', special_tokens)

    num_cpus = mp.cpu_count()

    vocab_p = f"{path}vocab.json"
    merges_p = f"{path}merges.txt"

    for split in ['valid']:
        process_split(
            split_name=split,
            input_path=f"{path}{split}.txt",
            output_path=f"{path}{split}.bin",
            vocab_path=vocab_p,
            merges_path=merges_p,
            special_tokens=special_tokens,
            num_proc=num_cpus
        )
