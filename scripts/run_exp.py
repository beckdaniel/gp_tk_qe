import numpy as np
import scipy as sp
import GPy
import nltk
import sys
import sklearn.preprocessing as pp
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.metrics import mean_squared_error as MSE
from scipy.stats import pearsonr as pearson
import os
from sklearn.svm import SVR
from sklearn.grid_search import GridSearchCV
import json


def save_parameters(m, param_file):
    param_names = m.parameter_names()
    param_values = m.param_array
    params = {name: val for name, val in zip(param_names, param_values)}
    with open(param_file, 'w') as f:
        json.dump(params, f)


###################################
def run_experiment(model_type):
    for fold in xrange(1):
        fold_dir = os.path.join(DATA_DIR, str(fold))
        train_data = np.loadtxt(os.path.join(fold_dir, 'train'), dtype=object, delimiter='\t')
        test_data = np.loadtxt(os.path.join(fold_dir, 'test'), dtype=object, delimiter='\t')

        X_trees_train = train_data[:SUB_TRAIN, 1:3]
        X_trees_test = test_data[:, 1:3]
        X_feats_train = np.array(train_data[:SUB_TRAIN, 3:], dtype=float)
        X_feats_test = np.array(test_data[:, 3:], dtype=float)
        Y_train = np.array(train_data[:SUB_TRAIN, 0:1], dtype=float)
        Y_test = np.array(test_data[:, 0:1], dtype=float)
    
        L = 0.5
        S = 1.0
        normalize = True

        en_tk = GPy.kern.SubsetTreeKernel(_lambda=L, _sigma=S, normalize=normalize, active_dims=[0], parallel=True, num_threads=NUM_THREADS)
        es_tk = GPy.kern.SubsetTreeKernel(_lambda=L, _sigma=S, normalize=normalize, active_dims=[1], parallel=True, num_threads=NUM_THREADS)
        en_tk['.*lambda.*'].constrain_bounded(1e-8,1)
        es_tk['.*lambda.*'].constrain_bounded(1e-8,1)
        en_tk['.*sigma.*'].constrain_bounded(1e-4,10)
        es_tk['.*sigma.*'].constrain_bounded(1e-4,10)

        rbf_iso = GPy.kern.Matern52(17, ARD=False)
        rbf_ard = GPy.kern.Matern52(17, ARD=True)
        rbf_iso['.*'].constrain_positive()
        rbf_ard['.*'].constrain_positive()

        if model_type == 1: # TK ADD
            k = en_tk + es_tk
        elif model_type == 2: # TK MUL
            k = en_tk * es_tk
        elif model_type == 3: # TK ADD, FIXED SIGMA
            en_tk['.*sigma.*'].constrain_fixed(1)
            es_tk['.*sigma.*'].constrain_fixed(1)
            k = en_tk + es_tk
        elif model_type == 4: # TK MUL, FIXED SIGMA
            en_tk['.*sigma.*'].constrain_fixed(1)
            es_tk['.*sigma.*'].constrain_fixed(1)
            k = en_tk * es_tk
        elif model_type >= 5: # RBF
            k = rbf_iso    
        
        if model_type <= 4:
            X_train = X_trees_train
            X_test = X_trees_test
        elif model_type == 5:
            X_train = X_feats_train
            X_test = X_feats_test
        elif model_type >= 6:
            X_train = np.concatenate((X_trees_train, X_feats_train), axis=1)
            X_test = np.concatenate((X_trees_test, X_feats_test), axis=1)

        if model_type >= 5:
            #m = GPy.models.GPRegression(X_feats_train, Y_train, kernel=rbf_iso)
            w = GPy.util.warping_functions.LogFunction()
            m = GPy.models.WarpedGP(X_feats_train, Y_train, kernel=rbf_iso, warping_function=w)
            m.optimize(max_iters=100, optimizer='lbfgs')
            iso_variance = m['rbf.variance'].copy()
            rbf_ard['lengthscale'] = m['rbf.lengthscale'].copy()
            noise_var = m['Gaussian_noise.variance'].copy()
            if model_type >= 6:
                #m = GPy.models.GPRegression(X_feats_train, Y_train, kernel=rbf_ard)
                w = GPy.util.warping_functions.LogFunction()
                m = GPy.models.WarpedGP(X_feats_train, Y_train, kernel=rbf_ard, warping_function=w)
                m['Gaussian_noise.variance'] = noise_var
                m['rbf.variance'] = iso_variance
                m.optimize(max_iters=100, optimizer='lbfgs')
                ard_variance = m['rbf.variance'].copy()
                ard_lengthscale = m['rbf.lengthscale'].copy()
                noise_var = m['Gaussian_noise.variance'].copy()

        
            
        if model_type == 5:
            k = rbf_ard
        else:
            rbf_ard = GPy.kern.Matern52(17, ARD=True, active_dims=range(2,19))
            if model_type == 6:
                k = en_tk + es_tk + rbf_ard
            elif model_type == 7:
                k = en_tk * es_tk * rbf_ard

        #m = GPy.models.GPRegression(X_train, Y_train, kernel=k)
        w = GPy.util.warping_functions.LogFunction()
        m = GPy.models.WarpedGP(X_train, Y_train, kernel=k, warping_function=w)
        if model_type >= 6:
            m['.*rbf.variance.*'].constrain_fixed(ard_variance)
            m['.*rbf.lengthscale.*'].constrain_fixed(ard_lengthscale)
            m['Gaussian_noise.variance'] = noise_var


        try:
            os.makedirs(os.path.join(PREDS_DIR, str(fold)))
        except:
            pass
            
        result_before = m.predict(X_test, median=True)
        np.savetxt(os.path.join(PREDS_DIR, str(fold), MODEL_NAMES[model_type] + '.before.pw'), result_before[0], fmt='%.5f')
        #result_before = (Y_scaler.inverse_transform(result_before[0]) * TEST_17[:,1:2], result_before[1])
        #np.savetxt(os.path.join(PREDS_DIR, str(fold), MODEL_NAMES[model_type] + '.before'), result_before[0], fmt='%.5f')
        np.savetxt(os.path.join(PREDS_DIR, str(fold), MODEL_NAMES[model_type] + '.before.vars'), result_before[1], fmt='%.5f')
        #np.savetxt(os.path.join(PREDS_DIR, str(fold), MODEL_NAMES[model_type] + '.before.params'), m['.*'], fmt='%.5f')
        #print repr(m['.*'])
        #import ipdb; ipdb.set_trace()
        save_parameters(m, os.path.join(PREDS_DIR, str(fold), MODEL_NAMES[model_type] + '.before.params'))
        

        mae_before =  MAE(result_before[0], Y_test)
        rmse_before =  np.sqrt(MSE(result_before[0], Y_test))
        pearson_before = pearson(result_before[0], Y_test)
        nlpd_before = -np.mean(m.log_predictive_density(X_test, Y_test))
        nll_before = m.log_likelihood()

        all_before_metrics = [mae_before, rmse_before, pearson_before[0], pearson_before[1], nlpd_before, nll_before]
        np.savetxt(os.path.join(PREDS_DIR, str(fold), MODEL_NAMES[model_type] + '.before.metrics'), all_before_metrics, fmt='%.5f')        

        print m
        m.optimize(max_iters=50, messages=True, optimizer='lbfgs')
        print m
        if model_type >= 5:
            print m['.*lengthscale.*']
        result_after = m.predict(X_test, median=True)

        np.savetxt(os.path.join(PREDS_DIR, str(fold), MODEL_NAMES[model_type] + '.after.pw'), result_after[0], fmt='%.5f')
        #result_after = (Y_scaler.inverse_transform(result_after[0]) * TEST_17[:,1:2], result_after[1])
        #np.savetxt(os.path.join(PREDS_DIR, str(fold), MODEL_NAMES[model_type] + '.after'), result_after[0], fmt='%.5f')
        np.savetxt(os.path.join(PREDS_DIR, str(fold), MODEL_NAMES[model_type] + '.after.vars'), result_after[1], fmt='%.5f')
        save_parameters(m, os.path.join(PREDS_DIR, str(fold), MODEL_NAMES[model_type] + '.after.params'))

        mae_after =  MAE(result_after[0], Y_test)
        rmse_after =  np.sqrt(MSE(result_after[0], Y_test))
        pearson_after = pearson(result_after[0], Y_test)
        nlpd_after = -np.mean(m.log_predictive_density(X_test, Y_test))
        nll_after = m.log_likelihood()

        all_after_metrics = [mae_after, rmse_after, pearson_after[0], pearson_after[1], nlpd_after, nll_after]
        np.savetxt(os.path.join(PREDS_DIR, str(fold), MODEL_NAMES[model_type] + '.after.metrics'), all_after_metrics, fmt='%.5f') 


##################################
def run_svm_experiment():
    tuned_parameters = [{'C': [1, 5, 10, 50, 100, 500, 1000],
                         'gamma': [1, 1e-1, 1e-2, 1e-3, 1e-4],
                         'epsilon': [0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1]}]

    cl = GridSearchCV(SVR(kernel='rbf'), tuned_parameters, n_jobs=NUM_THREADS)
    cl.fit(X_feats_train, Y_train.flatten())
    print cl.best_estimator_

    preds = cl.predict(X_feats_test)
    preds = Y_scaler.inverse_transform(preds)
    np.savetxt(os.path.join(PREDS_DIR, MODEL_NAMES[-1] + '.pw'), preds, fmt='%.5f')
    preds = preds * TEST_17[:,1]
    np.savetxt(os.path.join(PREDS_DIR, MODEL_NAMES[-1]), preds, fmt='%.5f')
    print preds.shape
    print Y_test.flatten().shape

    mae = MAE(preds, Y_test.flatten())
    rmse = np.sqrt(MSE(preds, Y_test.flatten()))

    train_preds = cl.predict(X_feats_train)
    train_preds = Y_scaler.inverse_transform(train_preds) * TRAIN_17[:,1]
    train_mae = MAE(train_preds, Y_train.flatten())
    train_rmse = np.sqrt(MSE(train_preds, Y_train.flatten()))

    return mae, rmse, mae, rmse, train_mae, train_rmse, train_mae, train_rmse

##################################


DATASET = sys.argv[1]
SUB_TRAIN = int(sys.argv[2])
NUM_THREADS = int(sys.argv[3])
MODELS = [int(m) for m in sys.argv[4].split(',')]
PREDS_DIR = sys.argv[5]
MODEL_NAMES = dict([(-1, 'SVM'),
                    (0, 'MEAN'),
                    (1, 'TK_ADD'),
                    (2, 'TK_MUL'),
                    (3, 'TK_ADD_FIXED_SIGMA'),
                    (4, 'TK_MUL_FIXED_SIGMA'),
                    (5, 'RBF'),
                    (6, 'RBF_TK_ADD'),
                    (7, 'RBF_TK_MUL')
                    ])


DATA_DIR = os.path.join('..', 'splits', DATASET)

try:
    os.makedirs(PREDS_DIR)
except:
    pass

results = {}
for model in MODELS:
    if model == -1:
        results[model] = run_svm_experiment()
    elif model > 0:
        results[model] = run_experiment(model)
    else:
        mean = np.mean(TRAIN_LABEL)
        preds = np.array([[mean] for elem in Y_test])
        mean_mae = MAE(preds, Y_test)
        mean_rmse = np.sqrt(MSE(preds, Y_test))
        train_preds = np.array([[mean] for elem in Y_train])
        train_mae = MAE(train_preds, Y_train)
        train_rmse = np.sqrt(MSE(train_preds, Y_train))
        results[model] = (mean_mae, mean_rmse, mean_mae, mean_rmse,
                          train_mae, train_rmse, train_mae, train_rmse)
