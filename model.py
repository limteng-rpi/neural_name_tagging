import torch
import torch.nn as nn
import torch.nn.init as I
import torch.nn.utils.rnn as R
import torch.nn.functional as F

import constant as C
from util import load_embedding_from_file
from module import Linear, LSTM, CRF, Linears, CharCNN, CharCNNFF


class LstmCNN(nn.Module):

    def __init__(self,
                 vocabs,
                 word_embed_file, word_embed_dim,
                 char_embed_dim, char_filters,
                 lstm_hidden_size,
                 lstm_dropout=0, feat_dropout=0,
                 ):
        # TODO: init function for saved model
        super(LstmCNN, self).__init__()

        self.vocabs = vocabs
        self.label_size = len(self.vocabs['label'])
        # Input features
        self.word_embed = load_embedding_from_file(word_embed_file,
                                                   word_embed_dim,
                                                   vocabs['token'],
                                                   vocabs['embed'],
                                                   vocabs['form'],
                                                   padding_idx=C.PAD_INDEX,
                                                   trainable=True)
        self.char_embed = CharCNN(len(vocabs['char']),
                                  char_embed_dim,
                                  char_filters)
        self.word_dim = self.word_embed.embedding_dim
        self.char_dim = self.char_embed.output_size
        self.feat_dim = self.char_dim + self.word_dim
        # Layers
        self.lstm = LSTM(input_size=self.feat_dim,
                         hidden_size=lstm_hidden_size,
                         batch_first=True,
                         bidirectional=True)
        self.output_linear = Linear(self.lstm.output_size,
                                    self.label_size)
        self.crf = CRF(vocabs['label'])
        self.feat_dropout = nn.Dropout(p=feat_dropout)
        self.lstm_dropout = nn.Dropout(p=lstm_dropout)

    def forward_nn(self, token_ids, char_ids, lens):
        batch_size, seq_len = token_ids.size()
        # word representation
        word_in = self.word_embed(token_ids)
        char_in = self.char_embed(char_ids)
        char_in = char_in.view(batch_size, seq_len, self.char_dim)
        feats = torch.cat([word_in, char_in], dim=2)
        feats = self.feat_dropout(feats)

        # LSTM layer
        lstm_in = R.pack_padded_sequence(feats, lens.tolist(), batch_first=True)
        lstm_out, _ = self.lstm(lstm_in)
        lstm_out, _ = R.pad_packed_sequence(lstm_out, batch_first=True)
        lstm_out = self.lstm_dropout(lstm_out)

        # output linear layer
        linear_out = self.output_linear(lstm_out)
        return linear_out

    def forward(self, token_ids, char_ids, lens, labels):
        logits = self.forward_nn(token_ids, char_ids, lens)
        logits = self.crf.pad_logits(logits)
        norm_score = self.crf.calc_norm_score(logits, lens)
        gold_score = self.crf.calc_gold_score(logits, labels, lens)
        loglik = gold_score - norm_score

        return loglik, logits

    def predict(self, token_ids, char_ids, lens):
        self.eval()
        logits = self.forward_nn(token_ids, char_ids, lens)
        logits = self.crf.pad_logits(logits)
        _scores, preds = self.crf.viterbi_decode(logits, lens)
        preds = preds.data.tolist()
        self.train()
        return preds


class LstmCNN_DFC(nn.Module):
    def __init__(self,
                 vocabs,
                 counts,
                 word_embed_file, word_embed_dim,
                 char_embed_dim, char_filters, char_feat_dim,
                 lstm_hidden_size,
                 lstm_dropout=0.5, feat_dropout=0.5, signal_dropout=0.5,
                 ctx_size=5
                 ):
        # TODO: init function for saved model
        assert char_feat_dim >= word_embed_dim
        super(LstmCNN_DFC, self).__init__()

        self.vocabs = vocabs
        self.label_size = len(self.vocabs['label'])
        # Input features
        self.word_embed = load_embedding_from_file(word_embed_file,
                                                   word_embed_dim,
                                                   vocabs['token'],
                                                   vocabs['embed'],
                                                   vocabs['form'],
                                                   padding_idx=C.PAD_INDEX,
                                                   trainable=True)
        self.char_embed = CharCNNFF(len(vocabs['char']),
                                    char_embed_dim,
                                    char_filters,
                                    char_feat_dim)
        self.word_dim = self.word_embed.embedding_dim
        self.char_dim = self.char_embed.output_size
        self.feat_dim = self.char_dim
        # Layers
        self.lstm = LSTM(input_size=self.feat_dim,
                         hidden_size=lstm_hidden_size,
                         batch_first=True,
                         bidirectional=True)
        self.output_linear = Linear(self.lstm.output_size,
                                    self.label_size)
        self.crf = CRF(vocabs['label'])
        self.feat_dropout = nn.Dropout(p=feat_dropout)
        self.lstm_dropout = nn.Dropout(p=lstm_dropout)
        self.signal_dropout = nn.Dropout(p=signal_dropout)

        # Word representation level gates
        self.word_gates = nn.ModuleList([
            nn.Linear(self.word_dim, self.word_dim),
            nn.Linear(self.word_dim, self.word_dim)])
        self.char_gates = nn.ModuleList([
            nn.Linear(self.word_dim, self.word_dim),
            nn.Linear(self.word_dim, self.word_dim)])
        self.signal_gates = nn.ModuleList([
            nn.Linear(self.signal_dim, self.word_dim),
            nn.Linear(self.signal_dim, self.word_dim)])