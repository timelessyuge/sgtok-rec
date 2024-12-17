def get_stats(ids):
    counts = {}
    for pair in zip(ids[:-1],ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts

def merge(ids, top_pair, idx):
    newids = []
    i = 0
    while i < len(ids):    
        # print(f"i: {i}")
        if i < len(ids) - 1 and ids[i] == top_pair[0] and ids[i+1] == top_pair[1]:
            newids.append(idx)
            i += 2
            # print(f"{pair} -> {idx}")
        else:
            newids.append(ids[i])
            i += 1

    return newids 

SPECIAL_TOKENS = {
    '<sos>': 10256,
    '<eos>': 10257,
    '<unk>': 10258,
    '<pad>': 10259,
    '<cls>': 10260,
    '<sep>': 10261,
    '<mask>': 10262,  # 10263 -> vocab
}

class Tokenizer:
    def __init__(self) -> None:
        self.merges = {}
        self.pattern = ""
        self.special_tokens = SPECIAL_TOKENS
        self.vocab = self._build_vocab()
        
    def bpe(self, text, vocab_size, verbose=False):
        assert vocab_size >= 256
        num_merges = vocab_size - 256
        text_bytes = text.encode("utf-8")
        ids = list(text_bytes)
        merges = {}
        
        for i in range(num_merges):
            stats = get_stats(ids)
            pair = max(stats, key=stats.get)
            idx = 256 + i
            ids = merge(ids, pair, idx)
            merges[pair] = idx
            self.vocab[idx] = self.vocab[pair[0]] + self.vocab[pair[1]]
            if verbose:
                print(f"合并 {i+1}/{num_merges}: {pair} -> {idx} (替代 {stats[pair]} 处 '{(self.vocab[idx]).decode("utf-8", errors="replace")}')")
        
        self.merges = merges
        
    
    def encode(self, text):
               
        text_bytes = text.encode("utf-8")
        ids = list(text_bytes)
        while len(ids) > 1:
            stats = get_stats(ids)
            pair = min(stats, key=lambda p: self.merges.get(p, float('inf')))
            if pair not in self.merges:
                break
            idx = self.merges[pair]
            ids = merge(ids,pair, idx)
        return ids

    def decode(self, ids):
        text_bytes = b"".join(self.vocab[idx] for idx in ids)
        text = text_bytes.decode("utf-8", errors="replace")
        return text
    
    def _build_vocab(self):
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for (p0, p1), idx in self.merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]
        for special, idx in self.special_tokens.items():
            vocab[idx] = special.encode("utf-8")
        return vocab

    def save(self, file_prefix):
        
        model_file = file_prefix + ".model"
        with open(model_file, 'w') as f:
            
            f.write("sgtok v1\n")
            f.write(f"{self.pattern}\n")
            
            f.write(f"{len(self.special_tokens)}\n")
            for special, idx in self.special_tokens.items():
                f.write(f"{special} {idx}\n")
            
            for idx1, idx2 in self.merges:
                f.write(f"{idx1} {idx2}\n")
        

    def load(self, model_file):       
        assert model_file.endswith(".model")
        merges = {}
        special_tokens = {}
        idx = 256
        with open(model_file, 'r', encoding="utf-8") as f:
            version = f.readline().strip()
            assert version == "sgtok v1"
            self.pattern = f.readline().strip()
            num_special = int(f.readline().strip())
            for _ in range(num_special):
                special, special_idx = f.readline().strip().split()
                special_tokens[special] = int(special_idx)
            for line in f:
                idx1, idx2 = map(int, line.split())
                merges[(idx1, idx2)] = idx
                idx += 1
        self.merges = merges
        self.special_tokens = special_tokens
        self.vocab = self._build_vocab()



if __name__ == "__main__":
    
    text = open('data/train.txt', encoding='utf-8').read()
    
    tokenizer = Tokenizer()
    
    import code
    code.interact(local=locals())
    
    tokenizer.bpe(text=text,vocab_size=10263, verbose=True)
    tokenizer.save('sgtokenizer')
    
    