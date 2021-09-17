import sys
import os
import numpy as np 
import tensorflow as tf
import pandas as pd

from scipy.signal import detrend
from sklearn.metrics import brier_score_loss

sys.path.append(os.path.expanduser('~/Documents/Hybrid/'))
from Hybrid.neuralnet import construct_modeldev_model, construct_climdev_model, preferred_loss, earlystop, BrierScore
from Hybrid.dataprep import prepare_full_set, test_trainval_split, filter_predictor_set, read_raw_predictand, twoclass_log_forecastprob, twoclass_logistic_regression_coefficients, multiclass_logistic_regression_coefficients, scale_time, scale_other_features

leadtimepool = 21 #list(range(20,22)) #[7,8,9,10,11,12,13] #[10,11,12,13,14,15] #[15,16,17,18,19,20,21] # From the longest leadtimepool is taken
target_region = 9
ndaythreshold = [4,7] #7 #[4,9] Switch to list for multiclass (n>2) predictions
#targetname = 'books_paper3-2_tg-ex-q0.75-21D_JJA_45r1_1D_0.01-t2m-grid-mean.csv' 
#targetname = 'books_paper3-2_tg-ex-q0.75-7D_JJA_45r1_1D_15-t2m-q095-adapted-mean.csv'
#targetname = 'books_paper3-2_tg-ex-q0.75-14D_JJA_45r1_1D_15-t2m-q095-adapted-mean.csv'
targetname = 'books_paper3-2_tg-ex-q0.75-21D_JJA_45r1_1D_15-t2m-q095-adapted-mean.csv'
predictors, forc, obs = prepare_full_set(targetname, ndaythreshold = ndaythreshold, predictand_cluster = target_region, leadtimepool = leadtimepool)
X_test, X_trainval, generator = test_trainval_split(predictors, crossval = True, nfolds = 4)
forc_test, forc_trainval, generator = test_trainval_split(forc, crossval = True, nfolds = 4)
obs_test, obs_trainval, generator = test_trainval_split(obs, crossval = True, nfolds = 4)
# Observation is already onehot encoded. Make a boolean last-class one for the benchmarks and the RF regressor
obs_trainval_bool = obs_trainval.iloc[:,-1].astype(bool)

# limiting X by j_measure
jfilter = filter_predictor_set(X_trainval, obs_trainval_bool, return_measures = False, nmost_important = 8, nbins=10)
# Also limiting by using a detrended target 
continuous_tg_name = 'books_paper3-1_tg-anom_JJA_45r1_14D-roll-mean_15-t2m-q095-adapted-mean.csv'
continuous_obs = read_raw_predictand(continuous_tg_name, clustid = 9, separation = leadtimepool)
continuous_obs = continuous_obs.reindex_like(X_trainval)
# Detrending
continuous_obs_detrended = pd.Series(detrend(continuous_obs.values), index = continuous_obs.index)
detrended_exceedence = continuous_obs_detrended > continuous_obs_detrended.quantile(0.75)

jfilter_det = filter_predictor_set(X_trainval.reindex(continuous_obs.index), detrended_exceedence, return_measures = False, nmost_important = 8, nbins=10)


dynamic_cols = X_trainval.loc[:,['swvl4','swvl13','z','sst','z-reg']].columns
dynamic_cols = dynamic_cols[~dynamic_cols.get_loc_level(-1, 'clustid')[0]] # Throw away the unclassified regime
final_trainval = X_trainval.loc[:,jfilter.columns.union(jfilter_det.columns).union(dynamic_cols)]
#final_trainval = X_trainval.loc[:,jfilter_det.columns.union(dynamic_cols)]
#final_trainval = X_trainval.loc[:,jfilter.columns.union(jfilter_det.columns)]
#final_trainval = X_trainval.loc[:,dynamic_cols] #jfilter_det
#final_trainval = jfilter_det

"""
Test the climdev keras 
"""
climprobkwargs, _, _ = multiclass_logistic_regression_coefficients(obs_trainval) # If multiclass will return the coeficients for all 
_, time_input, time_scaler, lr = twoclass_logistic_regression_coefficients(obs_trainval_bool, return_regressor = True) # Also fit a singleclass for the last category, this will be able to form the benchmark 
feature_input, feature_scaler = scale_other_features(final_trainval)
obs_input = obs_trainval.values


results = pd.DataFrame(np.nan, index = pd.MultiIndex.from_product([generator.groupids, ['train','val']], names = ['fold','part']), columns = ['crossentropy','accuracy','brier'])

for i, (trainind, valind) in enumerate(generator): # Entering the crossvalidation
    model = construct_climdev_model(n_classes = obs_trainval.shape[-1], n_hidden_layers= 0, n_features = final_trainval.shape[-1], climprobkwargs=climprobkwargs)
    #test = model([feature_input, time_input])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), #0.001 for no hidden layer
                loss=preferred_loss, #tf.keras.losses.CategoricalCrossentropy(from_logits=False),
                metrics=['accuracy',BrierScore()])
    history = model.fit(x = [feature_input[trainind,:],time_input[trainind]], 
            y = obs_input[trainind,:], 
            shuffle=True,
            batch_size=32,
            epochs=200, 
            validation_data=([feature_input[valind,:],time_input[valind]], obs_input[valind,:]),
            callbacks=[earlystop])

    results.loc[(i,'train'),:] = model.evaluate([feature_input[trainind,:],time_input[trainind]], obs_input[trainind,:])
    results.loc[(i,'val'),:] = model.evaluate([feature_input[valind,:],time_input[valind]], obs_input[valind,:])

generator.reset()

"""
Test the modeldev keras
"""
#raw_predictions = twoclass_log_forecastprob(forc_trainval)
#results2 = pd.DataFrame(np.nan, index = pd.MultiIndex.from_product([generator.groupids, ['train','val']], names = ['fold','part']), columns = ['crossentropy','accuracy','brier'])
#for i, (trainind, valind) in enumerate(generator): # Entering the crossvalidation
#    model = construct_modeldev_model(n_classes = 2, n_hidden_layers= 0, n_features = final_trainval.shape[-1])
#    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.05),
#                loss=preferred_loss, #tf.keras.losses.CategoricalCrossentropy(from_logits=False),
#                metrics=['accuracy',BrierScore()])
#    history = model.fit(x = [feature_input[trainind,:],raw_predictions[trainind]], 
#            y = obs_input[trainind,:], 
#            shuffle=True,
#            batch_size=32,
#            epochs=20, 
#            validation_data=([feature_input[valind,:],raw_predictions[valind]], obs_input[valind,:]), 
#            callbacks=[earlystop])
#    results2.loc[(i,'train'),:] = model.evaluate([feature_input[trainind,:],raw_predictions[trainind]], obs_input[trainind,:])
#    results2.loc[(i,'val'),:] = model.evaluate([feature_input[valind,:],raw_predictions[valind]], obs_input[valind,:])
#
#generator.reset()

"""
Test RF Hybrid model only empirical info and dynamical info of the intermediate variables
"""
#sys.path.append(os.path.expanduser('~/Documents/Weave/'))
#from Weave.models import HybridExceedenceModel, fit_predict_evaluate
#
#params = dict(fit_base_to_all_cv = True, max_depth = 5, n_estimators = 2500, min_samples_split = 30, max_features = 0.95, n_jobs = 5)
#evaluate_kwds = dict(scores = [brier_score_loss], score_names = ['brier'])
#
#results3 = pd.DataFrame(np.nan, index = pd.MultiIndex.from_product([generator.groupids, ['train','val']], names = ['fold','part']), columns = ['brier'])
#for i, (trainind, valind) in enumerate(generator): # Entering the crossvalidation
#    h = HybridExceedenceModel(**params)
#    scores = fit_predict_evaluate(model = h, X_in = final_trainval.iloc[trainind,:], y_in = obs_trainval.iloc[trainind],X_val = final_trainval.iloc[valind,:], y_val = obs_trainval.iloc[valind], evaluate_kwds = evaluate_kwds)
#    results3.loc[(i,'train'),:] = scores.loc['brier']
#    results3.loc[(i,'val'),:] = scores.loc['brier'] * scores.loc['brier_val/train']
#
#generator.reset()
#"""
#Test RF Hybrid model including forecast temperature
#"""
#forecasts = forc_trainval.copy()
#forecasts.name = 'pi'
#full_trainval = final_trainval.join(forecasts)
#results4 = pd.DataFrame(np.nan, index = pd.MultiIndex.from_product([generator.groupids, ['train','val']], names = ['fold','part']), columns = ['brier'])
#for i, (trainind, valind) in enumerate(generator): # Entering the crossvalidation
#    h = HybridExceedenceModel(**params)
#    scores = fit_predict_evaluate(model = h, X_in = full_trainval.iloc[trainind,:], y_in = obs_trainval.iloc[trainind],X_val = full_trainval.iloc[valind,:], y_val = obs_trainval.iloc[valind], evaluate_kwds = evaluate_kwds)
#    results4.loc[(i,'train'),:] = scores.loc['brier']
#    results4.loc[(i,'val'),:] = scores.loc['brier'] * scores.loc['brier_val/train']
#
#generator.reset()

"""
Benchmarks
Logistic regression is fitted on all train/validation data
"""
benchmarks = pd.DataFrame(np.nan, index = pd.MultiIndex.from_product([generator.groupids, ['val'], ['raw','trend']], names = ['fold','part','reference']), columns = ['brier'])
for i, (trainind, valind) in enumerate(generator): # Entering the crossvalidation
    benchmarks.loc[(i,'val','raw'),:] = brier_score_loss(y_true = obs_trainval_bool.iloc[valind], y_prob = forc_trainval.iloc[valind,-1])
    benchmarks.loc[(i,'val','trend'),:] = brier_score_loss(y_true = obs_trainval_bool.iloc[valind], y_prob = lr.predict_proba(time_input)[valind,-1])

bss = 1 - results.loc[(slice(None),'val'),'brier'] / benchmarks.loc[(slice(None),'val','trend'),'brier']
print(bss.values.round(3))
