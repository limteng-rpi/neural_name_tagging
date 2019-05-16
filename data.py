import re
import logging

import torch

import constant as C
from torch.utils.data import Dataset
from collections import Counter

logger = logging.getLogger()


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

    def numberize(self, vocabs, form_map):
        """Numberize the data set.
        :param vocabs: A dictionary of vocabularies.
        :param form_map: A mapping table from tokens in the data set to tokens
        in pre-trained word embeddings.
        """
        digit_pattern = re.compile('\d')

        token_vocab = vocabs['token']
        label_vocab = vocabs['label']
        char_vocab = vocabs['char']

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
            data.append([tokens_ids, char_ids, label_ids, tokens, labels])
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
                if len(chars) > max_seq_len:
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
