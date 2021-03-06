"""
Call signature: predictor_selection.py $CLIMDEV $NDAYTHRESHOLD
or...: predictors_selection.py $CLIMDEV $TIMEAGG $QUANTILE
"""
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
from Hybrid.neuralnet import DEFAULT_FIT, DEFAULT_COMPILE, ConstructorAndCompiler, construct_modeldev_model, construct_climdev_model
from Hybrid.dataprep import default_prep, extra_division
from Hybrid.optimization import multi_fit_single_eval, multi_fit_multi_eval

sys.path.append(os.path.expanduser('~/Documents/Weave/'))
from Weave.utils import collapse_restore_multiindex

crossval_scaling = True # Wether to do also minmax scaling in cv mode
do_climdev = sys.argv[1].lower() == 'true' # TRUE/FALSE Whether to do climdev or modeldev

if len(sys.argv[2:]) == 1: 
    print('assuming single argument is ndaythreshold')
    ndaythreshold = int(sys.argv[2])
    savename = f'tg-ex-q0.75-21D_ge{ndaythreshold}D_sep12-15'
else:
    print('assuming double argument is timeagg, quantile')
    timeagg = int(sys.argv[2])
    quantile = float(sys.argv[3])
    savename = f'tg-anom_JJA_45r1_{timeagg}D-roll-mean_q{quantile}_sep12-15'

basedir = Path('/scistor/ivm/jsn295/backup/')
#basedir = Path(f'/nobackup/users/straaten/')
basedir = basedir / f'{"clim" if do_climdev else ""}predsets/'
#basedir = basedir / f'{"clim" if do_climdev else ""}predsets3f/'
savedir = basedir / 'objective_balanced_cv/'


# With npreds = None all predictors are read, model needs to be reconfigured dynamically so no need to accept the default constructor
prepared_data, _ = default_prep(predictandname = savename, npreds = None, basedir = basedir, prepare_climdev = do_climdev ) #, division = extra_division)
# This prepared data has scaled trainval features, which we cannot scale again in cv mode, therefore replace with unscaled if neccesary
# NOTE: cv mode scaling/fitting does not happen for the climdev inputs (scaled time and climprobkwargs).
if crossval_scaling:
    feature_input = prepared_data.crossval.X_trainval.values
else:
    feature_input = prepared_data.neural.trainval_inputs[0] # These are prescaled. In same list as logforc
# Extract remaining neccesary data
generator = prepared_data.crossval.generator
extra_inp_trainval = prepared_data.neural.trainval_inputs[-1] # Best guess information stream, either log of forecast or time for the logistic regression
obsinp_trainval = prepared_data.neural.trainval_output

def score_model(training_data: Tuple[np.ndarray,np.ndarray], scoring_data: tuple = None) -> float:
    """
    Function to be called multiple times by PermutationImportance
    crossvalidation is optional inside this thing when training_data 
    contains the mixture of train and val
    training data gets manipulated by PermutationImportance so these contain only the features
    not the time_input or the logistic forc
    when crossvalidation we can disregard scoring_data (though it will be fed by permutation importance)
    also with SingleGenerator we can ignore it
    """
    g = copy(generator) # Such that when evaluated in parallel the iterators do not reset each other
    feature_trainval, y_trainval = training_data
    n_predictors = feature_trainval.shape[-1]
    # Dynamically adjust the complexity with the amount of inputs to the model
    if n_predictors == 1:
        nhidden = 1
        nhidden_nodes = 2 
    #elif n_predictors <= 4:
    #    nhidden = 1
    #    nhidden_nodes = 2
    else:
        nhidden = 1
        nhidden_nodes = 4

    construct_kwargs = dict(n_classes = y_trainval.shape[-1], 
            n_hidden_layers= nhidden, 
            n_features = n_predictors,
            n_hiddenlayer_nodes = nhidden_nodes)
    
    if do_climdev:
        construct_kwargs.update({'climprobkwargs':prepared_data.climate.climprobkwargs})
        construct_func = construct_climdev_model 
    else:
        construct_func = construct_modeldev_model
    constructor = ConstructorAndCompiler(construct_func, construct_kwargs, DEFAULT_COMPILE)
    
    DEFAULT_FIT.update({'epochs':40}) # Bit less epochs to speed up evaluations
    DEFAULT_FIT.update({'verbose':0}) # No verbosity when running on the cluster with stdout to file 

    scores = np.repeat(np.nan, n_eval)
    for i in range(n_eval):  # Possibly later in parallel
        score, histories = multi_fit_single_eval(constructor, X_trainval = (feature_trainval, extra_inp_trainval), y_trainval = y_trainval, generator = g, fit_kwargs = DEFAULT_FIT, return_predictions = False, scale_cv_mode = crossval_scaling) # Scores with RPS
        g.reset()
        scores[i] = score
    return scores.mean()


# THe n_jobs argument does not concern any bootstrapping, it concerns multiprocess evaluation. But tensorflow seems unable to fit in multiple processes.
# For bootstrapping and the score_model returning an array, it should be able to handle that with 'min' (namely 'argmin_of_mean' internally)

n_eval = 3 # Shuffling and random weight initialization lead to randomness, possibility to use multiple evaluations
depth = 20

newframe, oldlevels, olddtypes = collapse_restore_multiindex(prepared_data.crossval.X_test, axis = 1, inplace = False) # extracting names
result = sequential_forward_selection(training_data = (feature_input[:,:],obsinp_trainval), scoring_data = (feature_input[:,:],obsinp_trainval), scoring_fn = score_model, scoring_strategy = 'min', nimportant_vars = depth, variable_names=newframe.columns[:], njobs = 1)

singlepass = result.retrieve_singlepass()
multipass = result.retrieve_multipass()

# Writing out the selection and the whole predictor subset
singleresult = pd.DataFrame(singlepass, index = ['rank','rps'])
singleresult.to_csv(savedir / f'{savename}_single_d{depth}_b{n_eval}.csv')
multiresult = pd.DataFrame(multipass, index = ['rank','rps'])
multiresult.to_csv(savedir / f'{savename}_multi_d{depth}_b{n_eval}.csv')

multisubset = prepared_data.raw.predictors.iloc[:,newframe.columns.get_indexer(multiresult.columns)] # ordered by rank
multisubset.to_hdf(savedir / f'{savename}_multi_d{depth}_b{n_eval}_predictors.h5', key = 'input')

singlesubset = prepared_data.raw.predictors.iloc[:,newframe.columns.get_indexer(singleresult.columns)] # ordered by rank
singlesubset.to_hdf(savedir / f'{savename}_single_d{depth}_b{n_eval}_predictors.h5', key = 'input')
