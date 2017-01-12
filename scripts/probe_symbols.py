import numpy as np
import nltk
import sys
import os
from collections import defaultdict

SRC_TREES = np.loadtxt(sys.argv[1], dtype=object, delimiter='\t')
TGT_TREES = np.loadtxt(sys.argv[2], dtype=object, delimiter='\t')

src_symbols = defaultdict(int)
tgt_symbols = defaultdict(int)

for tree_s in SRC_TREES:
    t = nltk.tree.Tree.fromstring(tree_s)
    for pos in t.treepositions():
        t_pos = t[pos]
        if type(t_pos) != str:
            if type(t_pos[0]) != str:
                src_symbols[t_pos.label()] += 1

l = sorted(list(src_symbols.items()), key=lambda x: x[1], reverse=True)
print l[:10]
    
for tree_s in TGT_TREES:
    t = nltk.tree.Tree.fromstring(tree_s)
    for pos in t.treepositions():
        t_pos = t[pos]
        if type(t_pos) != str:
            if type(t_pos[0]) != str:
                tgt_symbols[t_pos.label()] += 1

l = sorted(list(tgt_symbols.items()), key=lambda x: x[1], reverse=True)
print l[:10]
