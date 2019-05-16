import re
import logging
import torch
import torch.nn as nn


logger = logging.getLogger()


def counter_to_vocab(counter, offset=0, pads=None, min_count=0):
    """Convert a counter to a vocabulary.
    :param count: A counter to convert.
    :param offset: Begin start offset.
    :param pads: A list of padding (str, index) pairs.
    :param min_count: Minimum count.
    """
    vocab = {}
    for token, freq in counter.items():
        if freq > min_count:
            vocab[token] = len(vocab) + offset
    if pads:
        for k, v in pads:
            vocab[k] = v

    return vocab


def merge_vocabs(vocabs, offset=0, pads=None):
    """Merge a list of vocabularies.
    :param vocabs: A list of vocabularies.
    :param offset: Index offset.
    :param pads: A list of special entries (e.g., PAD, SOS, EOS).
    """
    keys = set([k for v in vocabs for k in v.keys()])
    vocab = {key: idx for idx, key in enumerate(keys, offset)}
    if pads:
        for k, v in pads:
            vocab[k] = v
    return vocab


def build_embedding_vocab(path, skip_first=True):
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


def build_form_mapping(vocab: dict,
                          lower_case:bool = True,
                          zero_number:bool = True):
    form_mapping = {k: k for k, _v in vocab.items()}
    if not (lower_case or zero_number):
        return form_mapping

    digit_pattern = re.compile('\d')
    for k in vocab.keys():
        k_lower = k.lower()
        if lower_case:
            if k_lower not in form_mapping:
                form_mapping[k_lower] = k
        if zero_number:
            k_zero = re.sub(digit_pattern, '0', k_lower if lower_case else k)
            if k_zero not in form_mapping:
                form_mapping[k_zero] = k

    return form_mapping


def load_embedding_from_file(path,
                             embedding_dim,
                             vocab,
                             embed_vocab=None,
                             form_mapping=None,
                             padding_idx=None,
                             max_norm=None,
                             norm_type=2,
                             scale_grad_by_freq=False,
                             sparse=False,
                             trainable=True,
                             skip_first=True):
    """Load pre-trained embedding from file.
    :param path: Path to the embedding file.
    :param embedding_dim: Embedding dimension.
    :param vocab: Complete vocab. Some words in the complete vocab may be absent
    from the embedding vocab.
    :param embed_vocab: Embedding vocab.
    :param padding_idx: Padding index.
    :param sparse: Set this option to True may accelerate the training. Note
    that sparse gradient is not supported by all optimizers.
    """
    if embed_vocab is None:
        embed_vocab = build_embedding_vocab(path, skip_first=skip_first)
    if form_mapping is None:
        form_mapping = build_form_mapping(embed_vocab)

    logger.info('Loading embedding from file: {}'.format(path))
    weights = [[.0] * embedding_dim for _ in range(len(vocab))]
    with open(path, 'r', encoding='utf-8', errors='ignore') as r:
        if skip_first:
            r.readline()
        for line in r:
            try:
                segs = line.rstrip().split(' ')
                token = segs[0]
                if token in vocab:
                    weights[vocab[token]] = [float(i) for i in segs[1:]]
            except UnicodeDecodeError:
                pass

    # Fallback to lower case/all-zero number forms
    digit_pattern = re.compile('\d')
    for token, idx in vocab.items():
        if token not in embed_vocab:
            token_lower = token.lower()
            token_zero = re.sub(digit_pattern, '0', token_lower)
            if token_lower in form_mapping:
                weights[idx] = weights[vocab[form_mapping[token_lower]]]
            elif token_zero in form_mapping:
                weights[idx] = weights[vocab[form_mapping[token_zero]]]

    embed_mat = nn.Embedding(
        len(weights),
        embedding_dim,
        padding_idx=padding_idx,
        max_norm=max_norm,
        norm_type=norm_type,
        scale_grad_by_freq=scale_grad_by_freq,
        sparse=sparse,
        _weight=torch.FloatTensor(weights)
    )
    embed_mat.weight.requires_grad = trainable
    return embed_mat