import os
import glob
import torch

dir = '/data/m1/liny9/dynamic/model/lstmcnndfc'
conditions = {
    'ctx_size': 1,
    'signal_dropout': .2,
    'input': 'data/m1/liny9/dynamic/data/ontonotes/bn'
}

files = glob.glob(os.path.join(dir, '*', 'model.best.mdl'), recursive=True)
for file in files:
    state = torch.load(file)
    params = state['params']
    found = True
    for k, v in conditions.items():
        if params.get(k, None) != v:
            found = False
            break
    if found:
        print(file)
