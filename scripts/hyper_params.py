
import sys
import os
import numpy as np 
import tensorflow as tf
import pandas as pd

from scipy.signal import detrend
from sklearn.metrics import brier_score_loss

sys.path.append(os.path.expanduser('~/Documents/Hybrid/'))
from Hybrid.neuralnet import construct_modeldev_model, construct_climdev_model, preferred_loss, earlystop, BrierScore
from Hybrid.dataprep import prepare_full_set, test_trainval_split, filter_predictor_set, read_raw_predictand, twoclass_log_forecastprob, twoclass_logistic_regression_coefficients, scale_time, scale_other_features, one_hot_encoding

leadtimepool = [4,5,6,7] 
targetname = 'books_paper3-2_tg-ex-q0.75-7D_JJA_45r1_1D_15-t2m-q095-adapted-mean.csv'
predictors, forc, obs = prepare_full_set(targetname, ndaythreshold = 2, leadtimepool = leadtimepool)
X_test, X_trainval, generator = test_trainval_split(predictors, crossval = True, nfolds = 4)
forc_test, forc_trainval, generator = test_trainval_split(forc, crossval = True, nfolds = 4)
obs_test, obs_trainval, generator = test_trainval_split(obs, crossval = True, nfolds = 4)

# limiting X by j_measure
jfilter = filter_predictor_set(X_trainval, obs_trainval, return_measures = False, nmost_important = 10, nbins=10)

# Also limiting by using a detrended target
continuous_tg_name = 'books_paper3-1_tg-anom_JJA_45r1_7D-roll-mean_15-t2m-q095-adapted-mean.csv'
continuous_obs = read_raw_predictand(continuous_tg_name, clustid = 9, separation = leadtimepool)
continuous_obs = continuous_obs.reindex_like(X_trainval)
# Detrending
continuous_obs_detrended = pd.Series(detrend(continuous_obs.values), index = continuous_obs.index)
detrended_exceedence = continuous_obs_detrended > continuous_obs_detrended.quantile(0.75)

jfilter_det = filter_predictor_set(X_trainval, detrended_exceedence, return_measures = False, nmost_important = 10, nbins=10)

final_trainval = X_trainval.loc[:,jfilter.columns.union(jfilter_det.columns)]

# Data for the modeldev neural network
raw_predictions = twoclass_log_forecastprob(forc_trainval)
feature_input, feature_scaler = scale_other_features(final_trainval)
obs_input = one_hot_encoding(obs_trainval)

"""
Hyperparam optimization
"""

#parameters = [sherpa.Continuous(name='lr', range=[0.005, 0.1], scale='log'),
#              sherpa.Continuous(name='dropout', range=[0., 0.4]),
#              sherpa.Ordinal(name='batch_size', range=[16, 32, 64]),
#              sherpa.Discrete(name='num_hidden_units', range=[100, 300]),
#              sherpa.Choice(name='activation', range=['relu', 'elu', 'prelu'])]
#
#algorithm = sherpa.algorithms.RandomSearch(max_num_trials=150)
#study = sherpa.Study(parameters=parameters,
#                     algorithm=algorithm,
#                     lower_is_better=False)
#
#for trial in study:
#    model = init_model(trial.parameters)
#    for trainind, valind in generator:
#        training_error = model.fit(epochs=1)
#        validation_error = model.evaluate()
#        study.add_observation(trial=trial,
#                              iteration=iteration,
#                              objective=validation_error,
#                              context={'training_error': training_error})
#    generator.reset()
#    study.finalize(trial)
