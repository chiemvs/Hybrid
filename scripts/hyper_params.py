
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
Reading in pre-selected predictor sets 
and preconstructed targets
"""
savedir = Path('/nobackup/users/straaten/predsets/preselected/')
savename = 'tg-ex-q0.75-21D_ge7D_sep19-21_single'
#savename = 'tg-ex-q0.75-21D_ge7D_sep12-15' 
#savename = 'tg-anom_JJA_45r1_31D-roll-mean_sep19-21' 
predictors = pd.read_hdf(savedir / f'{savename}_predictors.h5', key = 'input')
forc = pd.read_hdf(savedir / f'{savename}_forc.h5', key = 'input')
obs = pd.read_hdf(savedir / f'{savename}_obs.h5', key = 'target')

X_test, X_trainval, generator = test_trainval_split(predictors, crossval = True, nfolds = 6)
forc_test, forc_trainval, generator = test_trainval_split(forc, crossval = True, nfolds = 6)
obs_test, obs_trainval, generator = test_trainval_split(obs, crossval = True, nfolds = 6)


climprobkwargs, time_input, time_scaler = multiclass_logistic_regression_coefficients(obs_trainval) # If multiclass will return the coeficients for all 
feature_input, feature_scaler = scale_other_features(X_trainval)
obs_input = obs_trainval.copy().values
raw_predictions = multiclass_log_forecastprob(forc_trainval)

"""
Hyperparam optimization
"""

parameters = [sherpa.Continuous(name='lr', range=[0.0003, 0.002]),
              sherpa.Discrete(name='earlystop_patience', range=[5, 20]),
              sherpa.Ordinal(name='batch_size', range=[16, 32, 64]),
              sherpa.Discrete(name='n_hidden_layers', range=[1,4]),
              sherpa.Discrete(name='n_hiddenlayer_nodes', range=[4,10])] #sherpa.Choice(name='activation', range=['relu', 'elu'])

algorithm = sherpa.algorithms.RandomSearch(max_num_trials=200)
study = sherpa.Study(parameters=parameters,
                     dashboard_port=8888,
                     disable_dashboard=True,
                     output_dir=f'/nobackup/users/straaten/predsets/preselected/{savename}/',
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
            callbacks = [earlystop(trial.parameters['earlystop_patience'])])

    # Noisy fitting, so do multiple evalutations, whose mean will converge with more evaluations
    n_eval = 8
    scores = np.repeat(np.nan, n_eval)
    for i in range(n_eval):  # Possibly later in parallel
        score, histories = multi_fit_single_eval(constructor, X_trainval = (feature_input, raw_predictions), y_trainval = obs_input, generator = generator, fit_kwargs = fit_kwargs, return_predictions = False)
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
