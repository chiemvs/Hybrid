import sys
import os
#import sherpa
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

"""
Test the climdev keras 
"""
climprobkwargs, time_input, time_scaler, lr = twoclass_logistic_regression_coefficients(obs_trainval, return_regressor = True) # Internally calls upon scale_time
feature_input, feature_scaler = scale_other_features(final_trainval)
obs_input = one_hot_encoding(obs_trainval)

results = pd.DataFrame(np.nan, index = pd.MultiIndex.from_product([generator.groupids, ['train','val']], names = ['fold','part']), columns = ['crossentropy','accuracy','brier'])

for i, (trainind, valind) in enumerate(generator): # Entering the crossvalidation
    model = construct_climdev_model(n_classes = 2, n_hidden_layers= 0, n_features = final_trainval.shape[-1], climprobkwargs=climprobkwargs)
    #test = model([feature_input, time_input])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.005),
                loss=preferred_loss, #tf.keras.losses.CategoricalCrossentropy(from_logits=False),
                metrics=['accuracy',BrierScore()])
    history = model.fit(x = [feature_input[trainind,:],time_input[trainind]], 
            y = obs_input[trainind,:], 
            epochs=20, 
            validation_data=([feature_input[valind,:],time_input[valind]], obs_input[valind,:]), 
            callbacks=[earlystop])

    results.loc[(i,'train'),:] = model.evaluate([feature_input[trainind,:],time_input[trainind]], obs_input[trainind,:])
    results.loc[(i,'val'),:] = model.evaluate([feature_input[valind,:],time_input[valind]], obs_input[valind,:])

generator.reset()
"""
Test the modeldev keras
"""
raw_predictions = twoclass_log_forecastprob(forc_trainval)
results2 = pd.DataFrame(np.nan, index = pd.MultiIndex.from_product([generator.groupids, ['train','val']], names = ['fold','part']), columns = ['crossentropy','accuracy','brier'])
for i, (trainind, valind) in enumerate(generator): # Entering the crossvalidation
    model = construct_modeldev_model(n_classes = 2, n_hidden_layers= 0, n_features = final_trainval.shape[-1])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.005),
                loss=preferred_loss, #tf.keras.losses.CategoricalCrossentropy(from_logits=False),
                metrics=['accuracy',BrierScore()])
    history = model.fit(x = [feature_input[trainind,:],raw_predictions[trainind]], 
            y = obs_input[trainind,:], 
            epochs=20, 
            validation_data=([feature_input[valind,:],raw_predictions[valind]], obs_input[valind,:]), 
            callbacks=[earlystop])
    results2.loc[(i,'train'),:] = model.evaluate([feature_input[trainind,:],raw_predictions[trainind]], obs_input[trainind,:])
    results2.loc[(i,'val'),:] = model.evaluate([feature_input[valind,:],raw_predictions[valind]], obs_input[valind,:])

generator.reset()

"""
Test RF Hybrid model only empirical info and dynamical info of the intermediate variables
"""
sys.path.append(os.path.expanduser('~/Documents/Weave/'))
from Weave.models import HybridExceedenceModel, fit_predict_evaluate

params = dict(fit_base_to_all_cv = True, max_depth = 5, n_estimators = 2500, min_samples_split = 30, max_features = 0.95, n_jobs = 5)
evaluate_kwds = dict(scores = [brier_score_loss], score_names = ['brier'])

results3 = pd.DataFrame(np.nan, index = pd.MultiIndex.from_product([generator.groupids, ['train','val']], names = ['fold','part']), columns = ['brier'])
for i, (trainind, valind) in enumerate(generator): # Entering the crossvalidation
    h = HybridExceedenceModel(**params)
    scores = fit_predict_evaluate(model = h, X_in = final_trainval.iloc[trainind,:], y_in = obs_trainval.iloc[trainind],X_val = final_trainval.iloc[valind,:], y_val = obs_trainval.iloc[valind], evaluate_kwds = evaluate_kwds)
    results3.loc[(i,'train'),:] = scores.loc['brier']
    results3.loc[(i,'val'),:] = scores.loc['brier'] * scores.loc['brier_val/train']

generator.reset()
"""
Test RF Hybrid model including forecast temperature
"""
forecasts = forc_trainval.copy()
forecasts.name = 'pi'
full_trainval = final_trainval.join(forecasts)
results4 = pd.DataFrame(np.nan, index = pd.MultiIndex.from_product([generator.groupids, ['train','val']], names = ['fold','part']), columns = ['brier'])
for i, (trainind, valind) in enumerate(generator): # Entering the crossvalidation
    h = HybridExceedenceModel(**params)
    scores = fit_predict_evaluate(model = h, X_in = full_trainval.iloc[trainind,:], y_in = obs_trainval.iloc[trainind],X_val = full_trainval.iloc[valind,:], y_val = obs_trainval.iloc[valind], evaluate_kwds = evaluate_kwds)
    results4.loc[(i,'train'),:] = scores.loc['brier']
    results4.loc[(i,'val'),:] = scores.loc['brier'] * scores.loc['brier_val/train']

generator.reset()

"""
Benchmarks
"""
benchmarks = pd.DataFrame(np.nan, index = pd.MultiIndex.from_product([generator.groupids, ['val'], ['raw','trend']], names = ['fold','part','reference']), columns = ['brier'])
for i, (trainind, valind) in enumerate(generator): # Entering the crossvalidation
    benchmarks.loc[(i,'val','raw'),:] = brier_score_loss(y_true = obs_trainval.iloc[valind], y_prob = forc_trainval.iloc[valind])
    benchmarks.loc[(i,'val','trend'),:] = brier_score_loss(y_true = obs_trainval.iloc[valind], y_prob = lr.predict_proba(time_input)[valind,-1])

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
