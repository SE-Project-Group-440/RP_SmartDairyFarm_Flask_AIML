from scripts.utils import (
    prepare_corpus,
    init_bpe_vocab,
    get_stats,
    merge_vocab,load_all_knowledge
)
import pickle

docs = load_all_knowledge("data/knowledge_base")
corpus = prepare_corpus(docs)

def train_bpe(corpus, num_merges=800):
    vocab = init_bpe_vocab(corpus)
    merges = []
    for _ in range(num_merges):
        pairs = get_stats(vocab)
        if not pairs:
            break
        best = max(pairs, key=pairs.get)
        vocab = merge_vocab(best, vocab)
        merges.append(best)
    return merges

bpe_merges = train_bpe(corpus)

bpe_vocab = {"<pad>": 0, "<unk>": 1}
idx = 2
for a, b in bpe_merges:
    token = a + b
    if token not in bpe_vocab:
        bpe_vocab[token] = idx
        idx += 1

with open("models/bpe_merges.pkl", "wb") as f:
    pickle.dump(bpe_merges, f)

with open("models/bpe_vocab.pkl", "wb") as f:
    pickle.dump(bpe_vocab, f)

print("BPE files saved")
