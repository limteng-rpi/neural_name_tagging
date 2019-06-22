# Dynamic Feature Composition for Name Tagging

Code for our ACL2019 paper _Reliability-aware Dynamic Feature Composition for Name Tagging_.

# Input Data Set Directory Structure
- <input_dir>
  - `embed.vocab.tsv`    (embedding vocab file, 1st column: token, 2nd column: index)
  - `embed.count.tsv`    (embedding token frequency file, 1st column: token, 2nd column: frequency)
  - `bc`
    - `train.tsv`        (training set)
    - `dev.tsv`          (development set)
    - `test.tsv`         (test set)
    - `token.vocab.tsv`  (token vocab file, 1st column: token, 2nd column: index)
    - `char.vocab.tsv`   (character vocab file: 1st column: character, 2nd column: index)
    - `label.vocab.tsv`  (label vocab file: 1st column: label, 2nd column: index)
  - `bn`
  - `mz`
  - `nw` 
  - `tc`
  - `wb`

Note:
- Other subsets have `train.tsv`, `dev.tsv`, `test.tsv`, `token.vocab.tsv`, `char.vocab.tsv`, and `label.vocab.tsv` in their directories.
- In our experiments, we generated `*.vocab.tsv` from a merged data set of all subsets.
- In our experiments, we use CoNLL format files generated from OntoNotes 5.0 with Pradhan et al.'s scripts, which can be found at https://cemantix.org/data/ontonotes.html.

# Pre-processing
The following functions in `proprocess.py` can be used to create vocab and frequency files.
- `build_all_vocabs` takes as input a list of CoNLL format files, and generate `{token,char,label}.vocab.tsv` in `output_dir`.
- `build_embed_vocab` takes a pre-trained embedding file as input and return the embedding vocab.
- `build_embed_token_count` takes a pre-trained embedding file as input and generate an embedding token frequency file.

# Train LSTM-CNN

```
python train_lstmcnn_all.py -d 4 -i <input_dir> -o <output_dir> -e <embedding_file>
  --embed_vocab <embedding_vocab_file> --char_dim 50 --seed <random_seed>
  -d <gpu_device>
```

This script train a model for each subset (which can be specified with the `--datasets` argument) and report within-subset (within-genre) and cross-subset (cross-genre) performance.

# Train LSTM-CNN with Dynamic Feature Composition

TBA

# Requirement
+ Python 3.5+
+ Pytorch 1.0

# Resources
+ We use the 100d case-sensitive word embedding in [Pre-trained Word Embeddings](http://www.limteng.com/research/2018/05/14/pretrained-word-embeddings.html)
