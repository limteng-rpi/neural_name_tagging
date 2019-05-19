from collections import defaultdict, Counter


def build_embed_pair_count(embed_train_file,
                           output_file,
                           window_size=5):
    token_counter = Counter()
    with open(embed_train_file, 'r', encoding='utf-8') as r:
        for line_num, line in enumerate(r, 1):
            if line_num % 1000000 == 0:
                print(line_num)
            tokens = [t for t in line.strip().split(' ') if t]
            token_counter.update(tokens)
    freq_tokens = {t for t, c in token_counter.items() if c > 10000}
    print('#freq: {}'.format(len(freq_tokens)))
    token_counter = None

    token_pairs = defaultdict(set)
    with open(embed_train_file, 'r', encoding='utf-8') as r:
        for line_num, line in enumerate(r, 1):
            if line_num % 1000000 == 0:
                print(line_num)
            tokens = [t for t in line.strip().split(' ') if t]
            if len(tokens) == 1:
                continue
            for i, token in enumerate(tokens):
                if token in freq_tokens:
                    continue
                for j in range(max(i - window_size, 0), i):
                    token_pairs[token].add(tokens[j])
                for j in range(i + 1, min(len(tokens), i + window_size + 1)):
                    token_pairs[token].add(tokens[j])
    token_pairs = [(t, len(c))
                   for t, c in token_pairs.items()]
    token_pairs.sort(key=lambda x: x[1], reverse=True)
    #
    with open(output_file, 'w', encoding='utf-8') as w:
        for t in freq_tokens:
            w.write('{}\t{}\n'.format(t, 99999))
        for t, c in token_pairs:
            w.write('{}\t{}\n'.format(t, c))


def build_embed_token_count(embed_train_file,
                            output_file):
    token_counter = Counter()
    with open(embed_train_file, 'r', encoding='utf-8') as r:
        for line_num, line in enumerate(r, 1):
            if line_num % 1000000 == 0:
                print(line_num)
            tokens = [t for t in line.strip().split(' ') if t]
            token_counter.update(tokens)
    token_counter = [(t, c) for t, c in token_counter.items()]
    token_counter.sort(key=lambda x: x[1], reverse=True)

    with open(output_file, 'w', encoding='utf-8') as w:
        for t, c in token_counter:
            w.write('{}\t{}\n'.format(t, c))

