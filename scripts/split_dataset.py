import numpy as np
import GPy
import sys
import os
from sklearn import cross_validation
from sklearn import preprocessing as pp
import gc


def filter_root(tree_array):
    for i,tree in enumerate(tree_array):
        tree_array[i][0] = tree[0][6:-2]
    return tree_array


def split_all_data():
    # READ DATA
    dataset_dir = os.path.join('..', 'data', DATASET)
    feats_file = os.path.join(dataset_dir, 'feats.17')
    src_trees_file = os.path.join(dataset_dir, 'source.trees')
    tgt_trees_file = os.path.join(dataset_dir, 'target.trees')
    labels_file = os.path.join(dataset_dir, 'time')

    feats = np.loadtxt(feats_file)
    labels = np.loadtxt(labels_file, ndmin=2)
    src_trees = np.loadtxt(src_trees_file, ndmin=2, dtype=object, delimiter='\t')
    tgt_trees = np.loadtxt(tgt_trees_file, ndmin=2, dtype=object, delimiter='\t')
    if DATASET == 'en-es':
        src_trees = filter_root(src_trees)
        tgt_trees = filter_root(tgt_trees)
    
    data = np.concatenate((labels, src_trees, tgt_trees, feats), axis=1)

    # time to PER
    data[:, 0] = data[:, 0] / data[:, 4]

    # SHUFFLE DATA
    np.random.seed(1000)
    np.random.shuffle(data)

    # BUILD FOLDER STRUCTURE
    dataset_dir = os.path.join(SPLIT_DIR, DATASET)
    try:
        os.makedirs(dataset_dir)
    except OSError:
        print "skipping folder creation"

    # SPLIT TRAIN/TEST AND SAVE
    fold_indices = cross_validation.KFold(data.shape[0], n_folds=10)
    for fold, index in enumerate(fold_indices):
        print index[0].shape
        train_data = data[index[0]]
        test_data = data[index[1]]

        train_feats = train_data[:, 3:]
        test_feats = test_data[:, 3:]
        scaler = pp.StandardScaler()
        scaler.fit(train_feats)
        train_feats = scaler.transform(train_feats)
        test_feats = scaler.transform(test_feats)

        train_data[:, 3:] = train_feats
        test_data[:, 3:] = test_feats

        fold_dir = os.path.join(dataset_dir, str(fold))
        try:
            os.makedirs(fold_dir)
        except OSError:
            print "skipping fold dir"
        np.savetxt(os.path.join(fold_dir, 'train'), train_data, fmt="%s", delimiter='\t')
        np.savetxt(os.path.join(fold_dir, 'test'), test_data, fmt="%s", delimiter='\t')

    
###########################    

DATASET = sys.argv[1]
SPLIT_DIR = os.path.join('..','splits')

# Generate the splits. Each split has mean-normalized features and
# pe-time per word in target segment.
split_all_data()
