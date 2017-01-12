import sys
import os
import numpy as np

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



for model in MODEL_NAMES:
    metrics_array = []
    before_array = []
    for fold in xrange(10):
        metrics = np.loadtxt(os.path.join(RESULTS_DIR, str(fold), model + '.after.metrics'))
        metrics_array.append(metrics)
        before = np.loadtxt(os.path.join(RESULTS_DIR, str(fold), model + '.before.metrics'))
        before_array.append(before)
    metrics_mean = np.nanmean(metrics_array, axis=0)
    before_mean = np.mean(before_array, axis=0)
    print model
    # print 'BEFORE: ',
    # print "MAE: %.4f" % before_mean[0],
    # print "RMSE: %.4f" % before_mean[1],
    # print "r: %.4f" % before_mean[2],
    # print "p: %.4f" % before_mean[3],
    # print "NLPD: %.4f" % before_mean[4],
    # print "NLL: %.4f" % before_mean[5]
    print 'AFTER: ',
    print "MAE: %.4f" % metrics_mean[0],
    print "RMSE: %.4f" % metrics_mean[1],
    print "r: %.4f" % metrics_mean[2],
    print "p: %.4f" % metrics_mean[3],
    print "NLPD: %.4f" % metrics_mean[4],
    print "NLL: %.4f" % metrics_mean[5]
        
