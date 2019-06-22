import re
import os
import json
import torch
import logging
import conlleval
import torch.nn as nn
import numpy as np
from collections import Counter, defaultdict
import constant as C

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
        if freq >= min_count:
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


def build_signal_embed(embed_counter, train_counter, token_vocab, form_mapping,
                       embed_scale_func=lambda x: np.tanh(.001 * x),
                       train_scale_func=lambda x: np.tanh(.1 * x)):
    """Building reliability signal embeddings.
    :param embed_counter: Embedding token or pair frequency.
    :param train_counter: Term frequency in the training set.
    :param token_vocab: Token vocabulary.
    :param form_mapping: Token form mapping (see build_form_mapping()).
    :param embed_scale_func: A scaling function.
    :param train_scale_func: A scaling function.
    """
    feat_size = 10
    # process counts
    embed_counter_scaled = {t: embed_scale_func(c)
                            for t, c in embed_counter.items()}
    train_counter_scaled = {t: train_scale_func(c)
                            for t, c in train_counter.items()}

    # build signal embeddings
    signal_embed = [[0] * feat_size for _ in range(len(token_vocab))]
    form_mapping_reversed = {}
    for k, vs in form_mapping.items():
        for v in vs:
            form_mapping_reversed[v] = k
    for token, token_idx in token_vocab.items():
        mapped_token = form_mapping_reversed.get(token, token)
        signal_embed[token_idx] = [
            # numeric signals
            embed_counter_scaled.get(mapped_token, 0),
            train_counter_scaled.get(token, 0),
            # binary signals
            1 if embed_counter.get(mapped_token, 0) < 5 else 0,
            1 if embed_counter.get(mapped_token, 0) < 10 else 0,
            1 if embed_counter.get(mapped_token, 0) < 100 else 0,
            1 if embed_counter.get(mapped_token, 0) < 1000 else 0,
            1 if embed_counter.get(mapped_token, 0) < 10000 else 0,
            1 if train_counter.get(token, 0) < 5 else 0,
            1 if train_counter.get(token, 0) < 10 else 0,
            1 if train_counter.get(token, 0) < 100 else 0,
        ]
    signal_embed = nn.Embedding.from_pretrained(torch.FloatTensor(signal_embed))
    return signal_embed


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


def load_vocab(path):
    vocab = {}
    with open(path, 'r', encoding='utf-8') as r:
        for line in r:
            token, idx = line.rstrip('\n').split('\t')
            vocab[token] = int(idx)
    return vocab


def calculate_lr(lr, current_step, total_step, min_lr=0):
    return min_lr + (lr - min_lr) * (1 - current_step / total_step)


def calculate_labeling_scores(results, report=True):
    outputs = []
    for p_b, g_b, t_b, l_b in results:
        for p_s, g_s, t_s, l_s in zip(p_b, g_b, t_b, l_b):
            p_s = p_s[:l_s]
            for p, g, t in zip(p_s, g_s, t_s):
                outputs.append('{} {} {}'.format(t, g, p))
            outputs.append('')
    counts = conlleval.evaluate(outputs)
    overall, by_type = conlleval.metrics(counts)
    if report:
        conlleval.report(counts)
    return (overall.fscore * 100.0, overall.prec * 100.0, overall.rec * 100.0)


def save_result_file(results, output_file, to_bio=False):
    def bioes_2_bio_tag(tag):
        if tag.startswith('S-'):
            tag = 'B-' + tag[2:]
        elif tag.startswith('E-'):
            tag = 'I-' + tag[2:]
        return tag

    with open(output_file, 'w', encoding='utf-8') as w:
        for p_b, g_b, t_b, l_b in results:
            for p_s, g_s, t_s, l_s in zip(p_b, g_b, t_b, l_b):
                p_s = p_s[:l_s]
                for p, g, t in zip(p_s, g_s, t_s):
                    if to_bio:
                        p = bioes_2_bio_tag(p)
                        g = bioes_2_bio_tag(g)
                    w.write('{}\t{}\t{}\n'.format(t, g, p))
                w.write('\n')
