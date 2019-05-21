import os
import re
import logging

import torch

import constant as C
from torch.utils.data import Dataset
from collections import Counter

logger = logging.getLogger()


def _bio_to_bioes(labels):
    label_len = len(labels)
    labels_bioes = []
    for idx, label in enumerate(labels):
        next_label = labels[idx + 1] if idx < label_len - 1 else 'O'
        if label == 'O':
            labels_bioes.append('O')
        elif label.startswith('B-'):
            if next_label.startswith('I-'):
                labels_bioes.append(label)
            else:
                labels_bioes.append('S-' + label[2:])
        else:
            if next_label.startswith('I-'):
                labels_bioes.append(label)
            else:
                labels_bioes.append('E-' + label[2:])
    return labels_bioes


def _apply_processor(inst, processor):
    for i, p in processor.items():
        inst[i] = p(inst[i])
    return inst

class ConllParser(object):

    def __init__(self,
                 fields, separator='\t',
                 skip_comment=False,
                 processor=None):
        self.fields = fields
        self.separator = separator
        self.skip_comment = skip_comment
        self.processor = processor

    def parse(self, path, *args, **kwargs):
        fields = kwargs.get('fields', self.fields)
        field_num = len(fields)
        separator = kwargs.get('separator', self.separator)
        skip_comment = kwargs.get('skip_comment', self.skip_comment)

        files = []
        if type(path) is list:
            files = path
        elif os.path.isdir(path):
            files = [f for f in os.listdir(path) if os.path.isfile((f))]
        elif os.path.isfile(path):
            files = [path]

        for file in files:
            with open(file, 'r', encoding='utf-8') as r:
                inst = [[] for _ in range(field_num)]
                for line in r:
                    if skip_comment and line.startswith('#'):
                        continue

                    line = line.rstrip('\n')
                    if line:
                        segs = line.split(separator)
                        for field_idx, field in enumerate(fields):
                            inst[field_idx].append(segs[field])
                    elif inst[0]:
                        if self.processor:
                            inst = _apply_processor(inst, self.processor)
                        yield inst
                        inst = [[] for _ in range(field_num)]
                if inst[0]:
                    if self.processor:
                        inst = _apply_processor(inst, self.processor)
                    yield inst



class NameTaggingDataset(Dataset):

    def __init__(self, path, parser, max_seq_len=-1, gpu=True, min_char_len=4):
        """
        :param path: Path to the data file.
        :param parser: A parser that read and process the data file.
        :param max_seq_len: Max sequence length (default=-1).
        """
        self.path = path
        self.parser = parser
        self.max_seq_len = max_seq_len
        self.data = []
        self.gpu = gpu
        self.min_char_len = min_char_len
        self.load()

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)

    def load(self):
        """Load data from the file"""
        logger.info('Loading data from {}'.format(self.path))
        self.data = [inst for inst in self.parser.parse(self.path)]

    @property
    def counters(self):
        """Get token, char, and label counters."""
        token_counter = Counter()
        char_counter = Counter()
        label_counter = Counter()
        for inst in self.data:
            tokens, labels = inst[0], inst[1]
            for token in tokens:
                for c in token:
                    char_counter[c] += 1
            token_counter.update(tokens)
            label_counter.update(labels)
        return token_counter, char_counter, label_counter

    @property
    def token_counter(self):
        token_counter = Counter()
        for inst in self.data:
            token_counter.update(inst[0])
        return token_counter

    def numberize(self, vocabs):
        """Numberize the data set.
        :param vocabs: A dictionary of vocabularies.
        :param form_map: A mapping table from tokens in the data set to tokens
        in pre-trained word embeddings.
        """
        digit_pattern = re.compile('\d')

        token_vocab = vocabs['token']
        label_vocab = vocabs['label']
        char_vocab = vocabs['char']
        form_map = vocabs['form']

        data = []
        for inst in self.data:
            tokens, labels = inst[0], inst[1]
            # numberize tokens
            tokens_ids = []
            for token in tokens:
                if token in token_vocab:
                    tokens_ids.append(token_vocab[token])
                else:
                    token_lower = token.lower()
                    token_zero = re.sub(digit_pattern, '0', token_lower)
                    if token_lower in form_map:
                        tokens_ids.append(token_vocab[form_map[token_lower]])
                    elif token_zero in form_map:
                        tokens_ids.append(token_vocab[form_map[token_zero]])
                    else:
                        tokens_ids.append(C.UNK_INDEX)
            # numberize characters and labels
            label_ids = [label_vocab[l] for l in labels]
            char_ids = [[char_vocab.get(c, C.UNK_INDEX) for c in t]
                        for t in tokens]
            if self.max_seq_len > 0:
                tokens_ids = tokens_ids[:self.max_seq_len]
                label_ids = label_ids[:self.max_seq_len]
                char_ids = char_ids[:self.max_seq_len]
            data.append((tokens_ids, char_ids, label_ids, tokens, labels))
        self.data = data

    def batch_processor(self, batch):
        pad = C.PAD_INDEX

        # sort instances in decreasing order of sequence lengths
        batch.sort(key=lambda x: len(x[0]), reverse=True)

        # sequence lengths
        seq_lens = [len(x[0]) for x in batch]
        max_seq_len = max(seq_lens)

        # character lengths
        max_char_len = self.min_char_len
        for seq in batch:
            for chars in seq[1]:
                if len(chars) > max_char_len:
                    max_char_len = len(chars)

        # padding instances
        batch_token_ids = []
        batch_char_ids = []
        batch_label_ids = []
        batch_tokens = []
        batch_labels = []
        for token_ids, char_ids, label_ids, tokens, labels in batch:
            seq_len = len(token_ids)
            pad_num = max_seq_len - seq_len
            batch_token_ids.append(token_ids + [pad] * pad_num)
            batch_char_ids.extend(
                # pad each word
                [x + [pad] * (max_char_len - len(x)) for x in char_ids] +
                # pad each sequence
                [[pad] * max_char_len for _ in range(pad_num)]
            )
            batch_label_ids.append(label_ids + [pad] * pad_num)
            batch_tokens.append(tokens)
            batch_labels.append(labels)

        if self.gpu:
            batch_token_ids = torch.cuda.LongTensor(batch_token_ids)
            batch_char_ids = torch.cuda.LongTensor(batch_char_ids)
            batch_label_ids = torch.cuda.LongTensor(batch_label_ids)
            seq_lens = torch.cuda.LongTensor(seq_lens)
        else:
            batch_token_ids = torch.LongTensor(batch_token_ids)
            batch_char_ids = torch.LongTensor(batch_char_ids)
            batch_label_ids = torch.LongTensor(batch_label_ids)
            seq_lens = torch.cuda.LongTensor(seq_lens)

        return (batch_token_ids, batch_char_ids, batch_label_ids, seq_lens,
                batch_tokens, batch_labels)
