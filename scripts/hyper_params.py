
import sys
import os
import sherpa
import numpy as np 
import pandas as pd
import tensorflow as tf

from pathlib import Path

sys.path.append(os.path.expanduser('~/Documents/Hybrid/'))
from Hybrid.neuralnet import DEFAULT_COMPILE, ConstructorAndCompiler, construct_modeldev_model, construct_climdev_model, earlystop
from Hybrid.dataprep import default_prep 
from Hybrid.optimization import multi_fit_single_eval

"""
Reading in pre-selected predictor sets (either sequential or jmeasure).
and preconstructed targets
"""
crossval_scaling = True # Wether to do also minmax scaling in cv mode
do_climdev = True # Whether to do climdev or modeldev
use_jmeasure = True
npreds = 4

#basedir = Path('/scistor/ivm/jsn295/backup/')
basedir = Path(f'/nobackup/users/straaten/')
basedir = basedir / f'{"clim" if do_climdev else ""}predsets/'
#ndaythreshold = 9
#savename = f'tg-ex-q0.75-21D_ge{ndaythreshold}D_sep12-15'
quantile = 0.5
timeagg = 31
predictandname = f'tg-anom_JJA_45r1_{timeagg}D-roll-mean_q{quantile}_sep12-15'

# With npreds = None all predictors are read, model needs to be reconfigured dynamically so no need to accept the default constructor
prepared_data, _ = default_prep(predictandname = predictandname, npreds = npreds, basedir = basedir, prepare_climdev = do_climdev, use_jmeasure = use_jmeasure)

generator = prepared_data.crossval.generator
# This prepared data has scaled trainval features, which we cannot scale again in cv mode, therefore replace with unscaled if neccesary
# NOTE: cv mode scaling/fitting does not happen for the climdev inputs (scaled time and climprobkwargs).
if crossval_scaling:
    feature_input = prepared_data.crossval.X_trainval.values
else:
    feature_input = prepared_data.neural.trainval_inputs[0] # These are prescaled. In same list as secondary input logforc (or time)
extra_inp_trainval = prepared_data.neural.trainval_inputs[-1] # Best guess information stream, either log of forecast or time for the logistic regression
savedir = basedir / f'hyperparams/{prepared_data.raw.predictor_name[:-14]}'
if savedir.exists():
    raise ValueError(f'hyperparam directory {savedir} already exists. Overwriting prevented. Check its content.')
else:
    pass
    savedir.mkdir()

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
    construct_kwargs = dict(n_classes = prepared_data.raw.obs.shape[-1], 
            n_hidden_layers= trial.parameters['n_hidden_layers'], 
            n_features = feature_input.shape[-1],
            n_hiddenlayer_nodes = trial.parameters['n_hiddenlayer_nodes'])
    
    if do_climdev:
        construct_kwargs.update({'climprobkwargs':prepared_data.climate.climprobkwargs})
        construct_func = construct_climdev_model 
    else:
        construct_func = construct_modeldev_model
    constructor = ConstructorAndCompiler(construct_func, construct_kwargs, DEFAULT_COMPILE)
    
    fit_kwargs = dict(batch_size = trial.parameters['batch_size'], 
            epochs = 200, 
            shuffle = True,
            callbacks = [earlystop(patience = trial.parameters['earlystop_patience'], monitor = 'val_loss')])

    # Noisy fitting, so do multiple evalutations, whose mean will converge with more evaluations
    n_eval = 8
    scores = np.repeat(np.nan, n_eval)
    for i in range(n_eval):  # Possibly later in parallel
        score, histories = multi_fit_single_eval(constructor, X_trainval = (feature_input, extra_inp_trainval), y_trainval = prepared_data.neural.trainval_output, generator = generator, fit_kwargs = fit_kwargs, return_predictions = False, scale_cv_mode = crossval_scaling)
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
