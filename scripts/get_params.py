import sys
import os
import numpy as np
import json
#from collections import defaultdict

MODEL_NAMES = dict([(1, 'TK_ADD'),
                    (2, 'TK_MUL'),
                    (3, 'TK_ADD_FIXED_SIGMA'),
                    (4, 'TK_MUL_FIXED_SIGMA'),
                    (5, 'MAT52'),
                    (6, 'MAT52_TK_ADD'),
                    (7, 'MAT52_TK_MUL')
                    ])

MODEL_NAMES = MODEL_NAMES.values()

RESULTS_DIR = sys.argv[1]

np.set_printoptions(suppress=True)


for model in MODEL_NAMES:
    params_array = {}
    for fold in xrange(10):
        with open(os.path.join(RESULTS_DIR, str(fold), model + '.after.params')) as f:
            params = json.load(f)
        for p in params:
            pval = params[p]
            if type(pval) == list:
                pval = np.array(pval)
            if p not in params_array:
                params_array[p] = pval
            else:
                params_array[p] += pval
    print model
    for p in params_array:
        print p, params_array[p] / 10
        
