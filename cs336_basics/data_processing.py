import json
import numpy as np
import cs336_basics.tokenizer as tokenizer
import cs336_basics.bpe as bpe

if __name__ == '__main__':
    path = "../datasets/tinystories_sample_"
    vocab_size = 10000

    special_tokens = ["<|endoftext|>"]

    vocab, merges = bpe.train_bpe(f'{path}train.txt', vocab_size, special_tokens)

    # vocab {int: bytes}
    vocab_to_json = {v.decode('utf-8', errors='replace'): k for k, v in vocab.items()}
    with open(f'{path}vocab.json', 'w', encoding='utf-8') as f:
        json.dump(vocab_to_json, f, indent=4, ensure_ascii=False)

    with open(f'{path}merges.txt', 'w', encoding='utf-8') as f:
        for p1, p2 in merges:
            f.write(f"{p1.decode('utf-8', errors='replace')} {p2.decode('utf-8', errors='replace')}\n")

    tok = tokenizer.Tokenizer(vocab, merges, special_tokens)

    for split in ['train', 'valid']:
        with open(f'{path}{split}.txt', 'r', encoding='utf-8') as f:
            text = f.read()
        ids = tok.encode(text)
        print(tok.decode(ids))
        ids_array = np.array(ids, dtype=np.uint16)
        ids_array.tofile(f'{path}{split}.bin')

