import os
import json

import time
import torch
import logging
import random
import logging
import numpy as np
from argparse import ArgumentParser

from torch.utils.data import DataLoader

import constant as C
from model import LstmCnn
from data import ConllParser, NameTaggingDataset
from util import calculate_labeling_scores, save_result_file

logging.basicConfig(level=logging.INFO,
                    format='%(message)s')
logger = logging.getLogger()

# parse commandline arguments
parser = ArgumentParser()
# i/o
parser.add_argument('-m', '--model', help='path to the model file')
parser.add_argument('-i', '--input', help='path to the test file')
parser.add_argument('-o', '--output', help='path to the output result file')
args = parser.parse_args()

use_gpu = torch.cuda.is_available()

logger.info('Load model from {}'.format(args.model))
state = torch.load(args.model)
params = state['params']
model = LstmCnn(vocabs=state['vocabs'],
                word_embed_file=None,
                word_embed_dim=params['word_dim'],
                char_embed_dim=params['char_dim'],
                char_filters=json.loads(params['char_filters']),
                char_feat_dim=params['char_feat_dim'],
                lstm_hidden_size=params['lstm_size'],
                parameters={'word_embed_num': 58002,
                            'word_embed_dim': 100},
                )
if use_gpu:
    model.cuda()

# data sets
logger.info('Load test data from {}'.format(args.input))
conll_parser = ConllParser([3, -1], processor={0: C.TOKEN_PROCESSOR})
test_set = NameTaggingDataset(args.input, conll_parser, gpu=use_gpu)
test_set.numberize(state['vocabs'])
label_itos = {i: s for s, i in state['vocabs']['label']}

results = []
for batch in DataLoader(test_set, batch_size=100, shuffle=False,
                        collate_fn=test_set.batch_processor):
    (token_ids, char_ids, label_ids, seq_lens,
     tokens, labels) = batch
    preds = model.predict(token_ids, char_ids, seq_lens)
    preds = [[label_itos[l] for l in ls] for ls in preds]
    results.append((preds, labels, tokens, seq_lens.tolist()))
fscore, prec, rec = calculate_labeling_scores(results)
logger.info('Test - P: {:.2f} R: {:.2f} F: {:.2f}'.format(
    prec, rec, fscore))
save_result_file(results, args.output)