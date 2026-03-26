import numpy as np
import random
from cs336_basics.tokenizer import Tokenizer


def verify_data(bin_path, vocab_json, merges_txt, special_tokens, sample_size=300):
    # 1. 加载你那套经过 GPT-2 映射加固的 Tokenizer
    print(f"正在加载 Tokenizer...")
    tokenizer = Tokenizer.from_files(vocab_json, merges_txt, special_tokens)

    # 2. 读取生成的二进制文件
    # 使用 uint16 是因为你的 vocab_size (10000) 小于 65535
    print(f"正在读取二进制文件: {bin_path}")
    ids_array = np.fromfile(bin_path, dtype=np.uint16)

    total_tokens = len(ids_array)
    print(f"文件总 Token 数: {total_tokens}")

    # 3. 随机抽取一段连续的 ID 进行解码
    if total_tokens > sample_size:
        start_idx = random.randint(0, total_tokens - sample_size)
        sample_ids = ids_array[start_idx: start_idx + sample_size].tolist()

        print("-" * 30)
        print(f"随机抽样片段 (Index {start_idx} to {start_idx + sample_size}):")

        # 核心步骤：解码
        decoded_text = tokenizer.decode(sample_ids)

        print("\n--- 解码内容 ---")
        print(decoded_text)
        print("--- 内容结束 ---\n")
    else:
        print("错误：二进制文件太短，无法抽样。")


if __name__ == "__main__":
    # 请根据你的实际路径修改
    PATH = "/root/autodl-tmp/cs336_assignment1/datasets/TinyStoriesV2-GPT4-"
    verify_data(
        bin_path=f"{PATH}valid.bin",
        vocab_json=f"{PATH}vocab.json",
        merges_txt=f"{PATH}merges.txt",
        special_tokens=["<|endoftext|>"]
    )
