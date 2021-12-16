
import sys
import os
import sherpa
import numpy as np 
import pandas as pd
import tensorflow as tf

from pathlib import Path

sys.path.append(os.path.expanduser('~/Documents/Hybrid/'))
from Hybrid.neuralnet import construct_modeldev_model, construct_climdev_model, earlystop, ConstructorAndCompiler
from Hybrid.dataprep import test_trainval_split, multiclass_logistic_regression_coefficients, scale_other_features, multiclass_log_forecastprob
from Hybrid.optimization import multi_fit_single_eval

"""
Reading in pre-selected predictor sets (either sequential or jmeasure).
and preconstructed targets
"""
#targetname = 'tg-ex-q0.75-21D_ge5D_sep12-15'
targetname = 'tg-anom_JJA_45r1_31D-roll-mean_q05_sep12-15'
selection = 'multi_d20_b3'
#selection = 'jmeasure-dyn'
basedir = Path('/nobackup/users/straaten')
savedir = basedir / f'hyperparams/{targetname}_{selection}'
if savedir.exists():
    raise ValueError(f'hyperparam directory {savedir} already exists. Overwriting prevented. Check its content.')
else:
    savedir.mkdir()
predictordir = basedir / 'predsets/objective_balanced_cv/' # Objectively sequential selected predictors
#predictordir = basedir / 'predsets/jmeasure/' # Objectively jmeasure selected predictors
for_obs_dir = basedir / 'predsets/full/' # contains corresponding forcasts and observations

predictors = pd.read_hdf(predictordir / f'{targetname}_{selection}_predictors.h5', key = 'input').iloc[:,:4]
forc = pd.read_hdf(for_obs_dir / f'{targetname}_forc.h5', key = 'input')
obs = pd.read_hdf(for_obs_dir / f'{targetname}_obs.h5', key = 'target')

crossval = True
balanced = True # Whether to use the balanced (hot dry years) version of crossvaldation. Folds are non-consecutive but still split by year. keyword ignored if crossval == False
crossval_scaling = True # Wether to do also minmax scaling in cv mode
nfolds = 3

X_test, X_trainval, generator = test_trainval_split(predictors, crossval = crossval, nfolds = nfolds, balanced = balanced)
forc_test, forc_trainval, generator = test_trainval_split(forc, crossval = crossval, nfolds = nfolds, balanced = balanced)
obs_test, obs_trainval, generator = test_trainval_split(obs, crossval = crossval, nfolds = nfolds, balanced = balanced)

climprobkwargs, time_input, time_scaler = multiclass_logistic_regression_coefficients(obs_trainval) # If multiclass will return the coeficients for all 
if crossval_scaling:
    feature_input = X_trainval.values
else:
    feature_input, feature_scaler = scale_other_features(X_trainval)
obs_input = obs_trainval.copy().values
raw_predictions = multiclass_log_forecastprob(forc_trainval)

"""
Hyperparam optimization
"""

parameters = [sherpa.Continuous(name='lr', range=[0.0003, 0.002]),
              sherpa.Discrete(name='earlystop_patience', range=[2, 11]),
              sherpa.Ordinal(name='batch_size', range=[16, 32, 64]),
              sherpa.Discrete(name='n_hidden_layers', range=[1,3]),
              sherpa.Discrete(name='n_hiddenlayer_nodes', range=[2,8])] #sherpa.Choice(name='activation', range=['relu', 'elu'])

algorithm = sherpa.algorithms.RandomSearch(max_num_trials=200)
study = sherpa.Study(parameters=parameters,
                     dashboard_port=8888,
                     disable_dashboard=True,
                     output_dir= savedir ,
                     algorithm=algorithm,
                     lower_is_better=True)


for trial in study:
    construct_kwargs = dict(n_classes = obs_trainval.shape[-1], 
            n_hidden_layers= trial.parameters['n_hidden_layers'], 
            n_features = X_trainval.shape[-1],
            n_hiddenlayer_nodes = trial.parameters['n_hiddenlayer_nodes'])
    
    compile_kwargs = dict(optimizer=tf.keras.optimizers.Adam(learning_rate=trial.parameters['lr']))

    constructor = ConstructorAndCompiler(construct_modeldev_model, construct_kwargs, compile_kwargs)

    fit_kwargs = dict(batch_size = trial.parameters['batch_size'], 
            epochs = 200, 
            shuffle = True,
            callbacks = [earlystop(trial.parameters['earlystop_patience'])])

    # Noisy fitting, so do multiple evalutations, whose mean will converge with more evaluations
    n_eval = 8
    scores = np.repeat(np.nan, n_eval)
    for i in range(n_eval):  # Possibly later in parallel
        score, histories = multi_fit_single_eval(constructor, X_trainval = (feature_input, raw_predictions), y_trainval = obs_input, generator = generator, fit_kwargs = fit_kwargs, return_predictions = False, scale_cv_mode = crossval_scaling)
        generator.reset()
        epochs = [len(h.epoch) for h in histories]
        scores[i] = score

        study.add_observation(trial=trial,
                              iteration=i,
                              objective=np.nanmean(scores),
                              context={'earliest_stop': min(epochs),
                                  'latest_stop':max(epochs)})
    study.finalize(trial)
study.save()
