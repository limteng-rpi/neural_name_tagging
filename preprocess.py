import os
import constant as C
from collections import defaultdict, Counter
from util import counter_to_vocab


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


def build_embed_vocab(path, skip_first=True):
    """Building a vocabulary from an embedding file.
    :param path: Path to the embedding file.
    """
    vocab = {}
    with open(path, 'r', encoding='utf-8', errors='ignore') as r:
        if skip_first:
            r.readline()
        for line in r:
            try:
                token = line.split(' ')[0].strip()
                if token:
                    vocab[token] = len(vocab)
            except UnicodeDecodeError:
                continue
    return vocab


def build_all_vocabs(files, output_dir, prefix=''):
    from data import ConllParser, NameTaggingDataset
    parser = ConllParser([3, -1], processor={0: C.TOKEN_PROCESSOR})
    token_counter, char_counter, label_counter = Counter(), Counter(), Counter()
    for file in files:
        dataset = NameTaggingDataset(file, parser)
        tc, cc, lc = dataset.counters
        token_counter.update(tc)
        char_counter.update(cc)
        label_counter.update(lc)
    token_vocab = counter_to_vocab(token_counter, offset=len(C.TOKEN_PADS), pads=C.TOKEN_PADS)
    char_vocab = counter_to_vocab(char_counter, offset=len(C.CHAR_PADS), pads=C.CHAR_PADS)
    label_vocab = counter_to_vocab(label_counter)

    token_vocab = [(t, c) for t, c in token_vocab.items()]
    char_vocab = [(t, c) for t, c in char_vocab.items()]
    label_vocab = [(t, c) for t, c in label_vocab.items()]

    with open(os.path.join(output_dir, '{}token.vocab.tsv'.format(prefix)),
              'w', encoding='utf-8') as w:
        for t, c in token_vocab:
            w.write('{}\t{}\n'.format(t, c))
    with open(os.path.join(output_dir, '{}char.vocab.tsv'.format(prefix)),
              'w', encoding='utf-8') as w:
        for t, c in char_vocab:
            w.write('{}\t{}\n'.format(t, c))
    with open(os.path.join(output_dir, '{}label.vocab.tsv'.format(prefix)),
              'w', encoding='utf-8') as w:
        for t, c in label_vocab:
            w.write('{}\t{}\n'.format(t, c))