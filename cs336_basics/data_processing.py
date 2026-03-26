import os
import json
import numpy as np
import multiprocessing as mp
from tqdm import tqdm
import cs336_basics.tokenizer as tokenizer
import cs336_basics.bpe as bpe
import cs336_basics.utils as utils

# --- 1. 把 worker 移到最外层 ---
_process_tokenizer = None
_process_eot_id = None


# --- 1. 修改后的子进程初始化 ---
def init_worker(tokenizer_params):
    """每个子进程启动时运行，获取具体的 ID"""
    global _process_tokenizer, _process_eot_id
    from cs336_basics.tokenizer import Tokenizer
    v_path, m_path, s_tokens = tokenizer_params
    _process_tokenizer = Tokenizer.from_files(v_path, m_path, s_tokens)

    # 因为 special_tokens 是 List[str]，我们直接调用 encode 来获取它的 ID
    # 注意：确保 encode 方法能处理特殊字符（通常会有 handle_special=True 之类的参数）
    # 或者如果你的 Tokenizer 有 special_token_to_id 字典，也可以直接取
    eot_string = "<|endoftext|>"
    _process_eot_id = _process_tokenizer.encode(eot_string)[0]
    # 打印一下确保没取错（可选）
    # print(f"Worker initialized. EOT ID: {_process_eot_id}")


def worker_func(doc_text):
    """处理一个完整的文档"""
    # 编码正文
    ids = _process_tokenizer.encode(doc_text)
    # 在末尾加上刚才取到的 EOT ID
    return ids + [_process_eot_id]


# --- 2. 修改后的 process_split (增加 tqdm 进度条) ---
def process_split(split_name, input_path, output_path, vocab_path, merges_path, special_tokens, num_proc):
    print(f"正在处理 {split_name} 数据...")
    init_params = (vocab_path, merges_path, special_tokens)

    def chunk_generator(file_path, delimiter='<|endoftext|>', batch_size=1000):
        """流式读取大文件，按 <|endoftext|> 分割文档"""
        with open(file_path, 'r', encoding='utf-8') as f:
            buffer = ""
            batch = []
            while True:
                # 每次读 4MB，防止 GB 级文件撑爆内存
                chunk = f.read(4 * 1024 * 1024)
                if not chunk:
                    if buffer.strip():
                        batch.append(buffer.strip())
                    if batch:
                        yield batch
                    break

                buffer += chunk
                while delimiter in buffer:
                    doc, buffer = buffer.split(delimiter, 1)
                    if doc.strip():
                        batch.append(doc.strip())
                    if len(batch) >= batch_size:
                        yield batch
                        batch = []

    all_ids = []

    # 估计文件大小用于显示进度（可选）
    file_size = os.path.getsize(input_path)

    with mp.Pool(num_proc, initializer=init_worker, initargs=(init_params,)) as pool:
        # 使用生成器配合 imap
        # batch_size 设为 1000 左右比较平衡
        for batch in chunk_generator(input_path, batch_size=1000):
            # chunksize=20 让每个进程一次处理 20 个文档，减少 IPC 通信开销
            results = pool.imap(worker_func, batch, chunksize=20)
            for res in results:
                all_ids.extend(res)

    print(f"正在转换 {split_name} 为 Numpy 并写入...")
    ids_array = np.array(all_ids, dtype=np.uint16)
    ids_array.tofile(output_path)
    print(f"{split_name} 处理完毕。")


if __name__ == '__main__':
    byte_encoder = utils.bytes_to_unicode()

    path = "/root/autodl-tmp/cs336_assignment1/datasets/TinyStoriesV2-GPT4-"
    special_tokens = ["<|endoftext|>"]

    vocab_size = 10000

    vocab, merges = bpe.train_bpe(f'{path}train.txt', vocab_size, special_tokens)

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

    num_cpus = mp.cpu_count()
    vocab_p = f"{path}vocab.json"
    merges_p = f"{path}merges.txt"

    for split in ['train', 'valid']:
        # 增加输入文件检查，防止路径错误
        in_p = f"{path}{split}.txt"
        out_p = f"{path}{split}.bin"
        if os.path.exists(in_p):
            process_split(
                split_name=split,
                input_path=in_p,
                output_path=out_p,
                vocab_path=vocab_p,
                merges_path=merges_p,
                special_tokens=special_tokens,
                num_proc=num_cpus
            )
        else:
            print(f"警告: 找不到输入文件 {in_p}")
