PAD = '<$PAD$>'
UNK = '<$UNK$>'
EOS = '<$EOS$>'
SOS = '<$SOS$>'
PAD_INDEX = 0       # Padding symbol index
UNK_INDEX = 1       # Unknown token index
SOS_INDEX = 2       # Start-of-sentence symbol index
EOS_INDEX = 3       # End-of-sentence symbol index
CHAR_PADS = [
    (PAD, PAD_INDEX),
    (UNK, UNK_INDEX),
]
TOKEN_PADS = [
    (PAD, PAD_INDEX),
    (UNK, UNK_INDEX),
    (SOS, SOS_INDEX),
    (EOS, EOS_INDEX),
]

TOKEN_REPLACE = {
    '-LRB-': '(',
    '-RRB-': ')',
    '-LSB-': '[',
    '-RSB-': ']',
    '-LCB-': '{',
    '-RCB-': '}',
    '``': '"',
    '\'\'': '"',
    '/.': '.',
    '/?': '?'
}

TOKEN_PROCESSOR = lambda x: [TOKEN_REPLACE.get(i, i) for i in x]