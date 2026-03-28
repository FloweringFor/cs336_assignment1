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
    print(f"🚀 正在并行分词: {split_name}")
    init_params = (vocab_path, merges_path, special_tokens)

    file_size = os.path.getsize(input_path)

    # 1. 优化后的流式生成器
    def doc_generator(file_path, pbar, delimiter='<|endoftext|>'):
        with open(file_path, 'r', encoding='utf-8') as f:
            buffer = ""
            while True:
                chunk = f.read(4 * 1024 * 1024)  # 4MB 缓冲区
                if not chunk:
                    if buffer.strip(): yield buffer.strip()
                    break

                # 更新进度条：读取了多少字节
                pbar.update(len(chunk.encode('utf-8')))

                buffer += chunk
                while delimiter in buffer:
                    doc, buffer = buffer.split(delimiter, 1)
                    if doc.strip():
                        yield doc.strip()

    # 2. 使用 tqdm 监控文件读取进度 (unit='B' 表示字节)
    with tqdm(total=file_size, unit='B', unit_scale=True, desc=f"Reading {split_name}") as pbar:
        with open(output_path, 'wb') as f_out:
            with mp.Pool(num_proc, initializer=init_worker, initargs=(init_params,)) as pool:
                # imap 配合 chunksize=100 保持 20 核满载
                results = pool.imap(worker_func, doc_generator(input_path, pbar), chunksize=100)

                chunk_ids = []
                total_tokens = 0

                for res in results:
                    if not res: continue
                    chunk_ids.extend(res)

                    # 每积攒 1M tokens (约 2MB uint16) 立即落盘，防止内存膨胀
                    if len(chunk_ids) >= 1_000_000:
                        np.array(chunk_ids, dtype=np.uint16).tofile(f_out)
                        total_tokens += len(chunk_ids)
                        chunk_ids = []

                # 写入末尾残余
                if chunk_ids:
                    np.array(chunk_ids, dtype=np.uint16).tofile(f_out)
                    total_tokens += len(chunk_ids)

    print(f"\n✅ {split_name} 处理完成！")
    print(f"📊 总计生成 Tokens: {total_tokens / 1e6:.2f} M")
    print(f"💾 输出文件路径: {output_path}")


if __name__ == '__main__':
    byte_encoder = utils.bytes_to_unicode()

    path = "/root/autodl-tmp/cs336_assignment1/datasets/owt_"
    special_tokens = ["<|endoftext|>"]

    vocab_size = 50257

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

"""
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
"""
