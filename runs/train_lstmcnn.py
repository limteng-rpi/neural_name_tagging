import os
import json
from collections import Counter

import tqdm
import time
import torch
import random
import logging
import numpy as np
from argparse import ArgumentParser

from torch.utils.data import DataLoader

from data import ConllParser, NameTaggingDataset
from model import LstmCNN
from util import counter_to_vocab, merge_vocabs, build_embedding_vocab,\
    build_form_mapping, build_signal_embed, load_vocab

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s-%(levelname)s: %(message)s')
logger = logging.getLogger()

# parse commandline arguments
parser = ArgumentParser()
# i/o
# parser.add_argument('--train', help='path to the train file')
# parser.add_argument('--dev', help='path to the dev file')
# parser.add_argument('--test', help='path to the test file')
parser.add_argument('-i', '--input', help='path to the input directory')
parser.add_argument('-o', '--output', help='path to the output directory')
# training parameters
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('-b', '--batch_size', type=int, default=10)
parser.add_argument('-m', '--max_epoch', type=int, default=20)
parser.add_argument('-s', '--seed', type=int, default=1111)
# model parameters
parser.add_argument('-e', '--embed')
parser.add_argument('--embed_vocab', default=None)
parser.add_argument('--char_dim', type=int, default=25)
parser.add_argument('--word_dim', type=int, default=100)
parser.add_argument('--char_filters', default='[[2,25],[3,25],[4,25]]')
parser.add_argument('--char_feat_dim', type=int, default=100)
parser.add_argument('--lstm_size', type=int, default=100)
parser.add_argument('--lstm_dropout', type=float, default=.5)
parser.add_argument('--feat_dropout', type=float, default=.5)
# parser.add_argument('--signal_dropout', type=float, default=.2)
parser.add_argument('--char_type', default='ffn',
                    help='ffn: feed-forward netword; hw: highway network')
# gpu
parser.add_argument('-d', '--device', type=int, default=0,
                    help='GPU device index')
args = parser.parse_args()
params = vars(args)

# timestamp
timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())

# output
output_dir = os.path.join(args.output, timestamp)
os.mkdir(output_dir)
best_model_file = os.path.join(output_dir, 'model.best.mdl')
# last_model_file = os.path.join(output_dir, 'model.last.mdl')
dev_result_file = os.path.join(output_dir, 'result.dev.bio')
test_result_file = os.path.join(output_dir, 'result.test.bio')
logger.info('Output directory: {}'.format(output_dir))

# deterministic behavior
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

# set gpu device
use_gpu = torch.cuda.is_available()
if use_gpu:
    torch.cuda.set_device(args.device)

# data sets
conll_parser = ConllParser([0, 1])
train_set = NameTaggingDataset(os.path.join(args.input, 'train.tsv'),
                               conll_parser, gpu=use_gpu)
dev_set = NameTaggingDataset(os.path.join(args.input, 'dev.tsv'),
                             conll_parser, gpu=use_gpu)
test_set = NameTaggingDataset(os.path.join(args.input, 'test.tsv'),
                              conll_parser, gpu=use_gpu)

# embedding
if args.embed_vocab:
    embed_vocab = load_vocab(args.embed_vocab)
else:
    embed_vocab = build_embedding_vocab(args.embed)

# vocabulary
token_vocab = load_vocab(os.path.join(args.input, 'token.vocab.tsv'))
char_vocab = load_vocab(os.path.join(args.input, 'char.vocab.tsv'))
label_vocab = load_vocab(os.path.join(args.input, 'label.vocab.tsv'))
train_token_counter = train_set.token_counter
vocabs = dict(token=token_vocab,
              char=char_vocab,
              label=label_vocab,
              embed=embed_vocab,
              form=build_form_mapping(token_vocab))
counters = dict(token=train_token_counter)

# numberize data set
train_set.numberize(vocabs)
dev_set.numberize(vocabs)
test_set.numberize(vocabs)

# create model
char_filters = json.loads(args.char_filters)
model = LstmCNN(vocabs=vocabs,
                word_embed_file=args.embed_file,
                word_embed_dim=args.word_dim,
                char_embed_dim=args.char_dim,
                char_filters=args.char_filters,
                lstm_hidden_size=args.lstm_size,
                lstm_dropout=args.lstm_dropout,
                feat_dropout=args.feat_dropout)
optimizer = torch.optim.Adam(filter(lambda x: x.requires_grad,
                                    model.parameters()),
                             lr=args.lr, weight_decay=.001)
if use_gpu:
    model.cuda()

# state
best_scores = {
    'dev': {'p': 0, 'r': 0, 'f': 0}, 'test': {'p': 0, 'r': 0, 'f': 0}}
state = dict(model=model.state_dict(),
             optimizer=optimizer.state_dict(),
             scores=best_scores,
             params=params,
             vocabs=vocabs,
             counters=counters)

# training
for epoch in range(args.max_epoch):
    logger.info('Epoch: {}'.format(epoch))
    epoch_loss = []

    for batch_idx, batch in enumerate(DataLoader(train_set,
                                                 batch_size=args.batch_size,
                                                 shuffle=True,
                                                 collate_fn=train_set.batch_processor)):
        optimizer.zero_grad()
        pass
