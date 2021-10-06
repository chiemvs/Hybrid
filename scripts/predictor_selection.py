import sys
import os
import numpy as np 
import pandas as pd
import tensorflow as tf

from copy import copy
from pathlib import Path
from typing import Tuple, List, Union
from PermutationImportance.sequential_selection import sequential_forward_selection

sys.path.append(os.path.expanduser('~/Documents/Hybrid/'))
from Hybrid.neuralnet import construct_modeldev_model, construct_climdev_model, earlystop, ConstructorAndCompiler
from Hybrid.dataprep import test_trainval_split, multiclass_logistic_regression_coefficients, scale_other_features, multiclass_log_forecastprob, GroupedGenerator
from Hybrid.optimization import multi_fit_single_eval

sys.path.append(os.path.expanduser('~/Documents/Weave/'))
from Weave.utils import collapse_restore_multiindex

savedir = Path('/nobackup/users/straaten/predsets/full/')
#savedir = Path('/scistor/ivm/jsn295/backup/predsets/full/')
savename = f'tg-ex-q0.75-21D_ge7D_sep19-21'
predictors = pd.read_hdf(savedir / f'{savename}_predictors.h5', key = 'input')
forc = pd.read_hdf(savedir / f'{savename}_forc.h5', key = 'input')
obs = pd.read_hdf(savedir / f'{savename}_obs.h5', key = 'target')

X_test, X_trainval, generator = test_trainval_split(predictors, crossval = False, nfolds = 6)
forc_test, forc_trainval, generator = test_trainval_split(forc, crossval = False, nfolds = 6)
obs_test, obs_trainval, generator = test_trainval_split(obs, crossval = False, nfolds = 6)


climprobkwargs, time_input, time_scaler = multiclass_logistic_regression_coefficients(obs_trainval) # If multiclass will return the coeficients for all 
feature_input, feature_scaler = scale_other_features(X_trainval)
obs_input = obs_trainval.copy().values
raw_predictions = multiclass_log_forecastprob(forc_trainval)


def score_model(training_data: Tuple[np.ndarray,np.ndarray], scoring_data: tuple = None) -> float:
    """
    Function to be called multiple times by PermutationImportance
    crossvalidation is optional inside this thing when training_data 
    contains the mixture of train and val
    training data gets manipulated by PermutationImportance so these contain only the features
    not the time_input or the logistic forc
    when crossvalidation we can disregard scoring_data
    also with SingleGenerator we can ignore it
    """
    g = copy(generator) # Such that when evaluated in parallel the iterators do not reset each other
    feature_trainval, y_trainval = training_data
    n_predictors = feature_trainval.shape[-1]
    # Scaling the complexity with the amount of inputs to the model
    if n_predictors == 1:
        nhidden = 0
        nhidden_nodes = 2
    elif n_predictors <= 4:
        nhidden = 1
        nhidden_nodes = 2
    else:
        nhidden = 2
        nhidden_nodes = 4

    construct_kwargs = dict(n_classes = y_trainval.shape[-1], 
            n_hidden_layers= nhidden, 
            n_features = n_predictors,
            n_hiddenlayer_nodes = nhidden_nodes)
    
    compile_kwargs = dict(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
    
    constructor = ConstructorAndCompiler(construct_modeldev_model, construct_kwargs, compile_kwargs)
    
    fit_kwargs = dict(batch_size = 32, 
            epochs = 200, 
            shuffle = True,
            callbacks = [earlystop(10)])

    score, histories = multi_fit_single_eval(constructor, X_trainval = (feature_trainval, raw_predictions), y_trainval = y_trainval, generator = g, fit_kwargs = fit_kwargs, return_predictions = False) # Scores with RPS
    # for SingleGenerator nans are written in multi_fit_single_eval when doing only one fold. Therefore handled in the single_eval
    return score

#test2 = score_model((feature_input[:,:10],obs_input))

# THe n_jobs argument does not concern any bootstrapping, it concerns multiprocess evaluation. But tensorflow seems unable to fit in multiple processes.
# For bootstrapping and the score_model returning an array, it should be able to handle that with 'min' (namely 'argmin_of_mean' internally)

depth = 15
newframe, oldlevels, olddtypes = collapse_restore_multiindex(X_test, axis = 1, inplace = False) # extracting names
result = sequential_forward_selection(training_data = (feature_input[:,:5],obs_input), scoring_data = (feature_input[:,:5],obs_input), scoring_fn = score_model, scoring_strategy = 'min', nimportant_vars = depth, variable_names=newframe.columns[:5], njobs = 1)

singlepass = result.retrieve_singlepass()
multipass = result.retrieve_multipass()

singleresult = pd.DataFrame(singlepass, index = ['rank','rps'])
singleresult.to_csv(savedir / f'{savename}_singlepass.csv')
multiresult = pd.DataFrame(multipass, index = ['rank','rps'])
multiresult.to_csv(savedir / f'{savename}_multipass_depth{depth}.csv')
