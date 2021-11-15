import sys
import os
import numpy as np 
import xarray as xr
import tensorflow as tf
import pandas as pd

from pathlib import Path
from scipy.signal import detrend
from sklearn.metrics import brier_score_loss
from sklearn.linear_model import LogisticRegression, LinearRegression

sys.path.append(os.path.expanduser('~/Documents/Hybrid/'))
from Hybrid.neuralnet import construct_modeldev_model, construct_climdev_model, reducelr, earlystop, BrierScore, ConstructorAndCompiler
from Hybrid.optimization import multi_fit_multi_eval, multi_fit_single_eval, ranked_prob_score
from Hybrid.dataprep import test_trainval_split, filter_predictor_set, multiclass_log_forecastprob, singleclass_regression, multiclass_logistic_regression_coefficients, scale_other_features, generate_balanced_kfold
from Hybrid.dataloading import prepare_full_set, read_raw_predictand, read_tganom_predictand, read_raw_predictor_regimes 

leadtimepool = list(range(12,16)) #list(range(12,16)) #list(range(19,22)) #[7,8,9,10,11,12,13] #[10,11,12,13,14,15] #[15,16,17,18,19,20,21] # From the longest leadtimepool is taken
target_region = 9 
ndaythreshold = 7 #[3,7] #7 #[4,9] Switch to list for multiclass (n>2) predictions
focus_class = -1 # Index of the class to be scored and benchmarked through bss
multi_eval = True # Single aggregated score or one per fold
preload = True
crossval = True
balanced = True # Whether to use the balanced (hot dry years) version of crossvaldation. Folds are non-consecutive but still split by year. keyword ignored if crossval == False
crossval_scaling = True # Wether to do also minmax scaling in cv mode
nfolds = 3
#targetname = 'books_paper3-2_tg-ex-q0.75-21D_JJA_45r1_1D_0.01-t2m-grid-mean.csv' 
#targetname = 'books_paper3-2_tg-ex-q0.75-7D_JJA_45r1_1D_15-t2m-q095-adapted-mean.csv'
#targetname = 'books_paper3-2_tg-ex-q0.75-14D_JJA_45r1_1D_15-t2m-q095-adapted-mean.csv'
targetname = 'books_paper3-2_tg-ex-q0.75-21D_JJA_45r1_1D_15-t2m-q095-adapted-mean.csv'
predictors, forc, obs = prepare_full_set(targetname, ndaythreshold = ndaythreshold, predictand_cluster = target_region, leadtimepool = leadtimepool)
if preload: # For instance a predictor set coming from 
    loadpath = '/nobackup/users/straaten/predsets/objective_balanced_cv/tg-anom_JJA_45r1_31D-roll-mean_sep12-15_multi_d20_b3_predictors.h5'
    predictors = pd.read_hdf(loadpath, key = 'input').iloc[:,:15]


"""
Predictand replacement with tg-anom
Quite involved because thresholds need to be matched
"""
#tganom_name = 'books_paper3-1_tg-anom_JJA_45r1_31D-roll-mean_15-t2m-q095-adapted-mean.csv'
#climname = 'tg-anom_clim_1998-06-07_2019-10-31_31D-roll-mean_15-t2m-q095-adapted-mean_5_5_q0.75'
#modelclimname = 'tg-anom_45r1_1998-06-07_2019-08-31_31D-roll-mean_15-t2m-q095-adapted-mean_5_5_q0.75'
#
#tgobs, tgforc = read_tganom_predictand(booksname = tganom_name, clustid = target_region, separation = leadtimepool, climname = climname, modelclimname = modelclimname) 
#forc, obs = tgforc.loc[forc.index,:], tgobs.loc[forc.index,:]


"""
Predictand replacement with regimes
"""
#regimename = 'books_paper3-4-regimes_z-anom_JJA_45r1_21D-frequency_ids.csv'
#regforc, regobs = read_raw_predictor_regimes(booksname = regimename, clustid = slice(None), separation = leadtimepool, observation_too = True)
#forc, obs = regforc.loc[forc.index,:], regobs.loc[forc.index,:]

"""
Cross validation
"""
X_test, X_trainval, generator = test_trainval_split(predictors, crossval = crossval, nfolds = nfolds, balanced = balanced)
forc_test, forc_trainval, generator = test_trainval_split(forc, crossval = crossval, nfolds = nfolds, balanced = balanced)
obs_test, obs_trainval, generator = test_trainval_split(obs, crossval = crossval, nfolds = nfolds, balanced = balanced)
# Observation is already onehot encoded. Make a boolean last-class one for the benchmarks and the RF regressor
obs_trainval_bool = obs_trainval.iloc[:,focus_class].astype(bool)

if not preload:
    """
    Semi-objective predictor selection
    """
    ## limiting X by j_measure
    #jfilter = filter_predictor_set(X_trainval, obs_trainval_bool, return_measures = False, nmost_important = 8, nbins=10)
    ## Also limiting by using a detrended target 
    #continuous_tg_name = 'books_paper3-1_tg-anom_JJA_45r1_14D-roll-mean_15-t2m-q095-adapted-mean.csv'
    #continuous_obs = read_raw_predictand(continuous_tg_name, clustid = 9, separation = leadtimepool)
    #continuous_obs = continuous_obs.reindex_like(X_trainval)
    ## Detrending
    #continuous_obs_detrended = pd.Series(detrend(continuous_obs.values), index = continuous_obs.index)
    #detrended_exceedence = continuous_obs_detrended > continuous_obs_detrended.quantile(0.75)
    #
    #jfilter_det = filter_predictor_set(X_trainval.reindex(continuous_obs.index), detrended_exceedence, return_measures = False, nmost_important = 8, nbins=10)
    #
    #
    #dynamic_cols = X_trainval.loc[:,['swvl4','swvl13','z','sst','z-reg']].columns
    #dynamic_cols = dynamic_cols[~dynamic_cols.get_loc_level(-1, 'clustid')[0]] # Throw away the unclassified regime
    #dynamic_cols = dynamic_cols[~dynamic_cols.get_loc_level('z-reg', 'variable')[0]] # Throw away all regimes
    #final_trainval = X_trainval.loc[:,jfilter.columns.union(jfilter_det.columns).union(dynamic_cols)]
    #final_trainval = X_trainval.loc[:,jfilter_det.columns.union(dynamic_cols)]
    #final_trainval = X_trainval.loc[:,jfilter.columns.union(dynamic_cols)]
    #final_trainval = X_trainval.loc[:,jfilter.columns.union(jfilter_det.columns)]
    #final_trainval = X_trainval.loc[:,dynamic_cols] #jfilter_det
    #final_trainval = jfilter_det
    #final_trainval = X_trainval.drop(dynamic_cols, axis = 1)
    final_trainval = X_trainval.loc[:,~X_trainval.columns.get_loc_level(-1, 'clustid')[0]] # Throw away the unclassified
    #final_trainval = X_trainval.loc[:,~X_trainval.columns.get_loc_level('z-reg', 'variable')[0]] # Throw away all regimes
    
    """
    optionally Saving full set for later
    """
    #savedir = Path('/nobackup/users/straaten/predsets/full/')
    ##savename = f'tg-ex-q0.75-21D_ge{ndaythreshold}D_sep{leadtimepool[0]}-{leadtimepool[-1]}'
    ##savename = f'tg-anom_JJA_45r1_31D-roll-mean_sep{leadtimepool[0]}-{leadtimepool[-1]}'
    #savename = f'regimes_z-anom_JJA_45r1_21D-frequency_sep{leadtimepool[0]}-{leadtimepool[-1]}'
    #predictors.loc[:,final_trainval.columns].to_hdf(savedir / f'{savename}_predictors.h5', key = 'input')
    #forc.to_hdf(savedir / f'{savename}_forc.h5', key = 'input')
    #obs.to_hdf(savedir / f'{savename}_obs.h5', key = 'target')
else:
    final_trainval = X_trainval

"""
Preparation of ANN input and a benchmark
"""
climprobkwargs, _, _ = multiclass_logistic_regression_coefficients(obs_trainval) # If multiclass will return the coeficients for all 
time_input, time_scaler, lr = singleclass_regression(obs_trainval_bool, regressor = LogisticRegression ) # fit a singleclass for the last category, this will be able to form the benchmark 
#time_input, time_scaler, lr = singleclass_regression(obs_trainval_bool, regressor = LinearRegression) # fit a singleclass for the last category, this will be able to form the benchmark 
feature_input, feature_scaler = scale_other_features(final_trainval) 
if crossval_scaling:
    feature_input = final_trainval.values # Override
obs_input = obs_trainval.copy().values

"""
Test the climdev keras 
"""
#construct_kwargs = dict(n_classes = obs_trainval.shape[-1], 
#        n_hidden_layers= 1, 
#        n_features = final_trainval.shape[-1],
#        climprobkwargs = climprobkwargs)
#
#compile_kwargs = dict(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
#        metrics = ['accuracy',BrierScore(class_index = focus_class)])
#
#constructor = ConstructorAndCompiler(construct_climdev_model, construct_kwargs, compile_kwargs)
#
#fit_kwargs = dict(batch_size = 32, epochs = 200, callbacks = [earlystop(10)])
##fit_kwargs = dict(batch_size = 32, epochs = 15)
#
#if multi_eval:
#    results = multi_fit_multi_eval(constructor, X_trainval = (feature_input, time_input), y_trainval = obs_input, generator = generator, fit_kwargs = fit_kwargs)
#    results.columns = ['crossentropy','accuracy','brier'] # coould potentially also be inside the multi_eval, but difficult to get names from the mixture of strings and other
#else:
#    score, predictions = multi_fit_single_eval(constructor, X_trainval = (feature_input, time_input), y_trainval = obs_input, generator = generator, fit_kwargs = fit_kwargs, return_predictions = True)
#
#generator.reset()

"""
Test the modeldev keras
"""
raw_predictions = multiclass_log_forecastprob(forc_trainval)

construct_kwargs = dict(n_classes = obs_trainval.shape[-1], 
        n_hidden_layers= 2, 
        n_features = final_trainval.shape[-1],
        n_hiddenlayer_nodes = 4)

compile_kwargs = dict(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
        metrics = ['accuracy',BrierScore(class_index = focus_class)])

constructor = ConstructorAndCompiler(construct_modeldev_model, construct_kwargs, compile_kwargs)

fit_kwargs = dict(batch_size = 32, epochs = 200, shuffle = True, callbacks = [earlystop(patience = 10, monitor = 'val_loss')])

if multi_eval:
    results = multi_fit_multi_eval(constructor, X_trainval = (feature_input, raw_predictions), y_trainval = obs_input, generator = generator, fit_kwargs = fit_kwargs, scale_cv_mode = crossval_scaling)
    results.columns = ['crossentropy','accuracy','brier'] # coould potentially also be inside the multi_eval, but difficult to get names from the mixture of strings and other
else:
    score, predictions = multi_fit_single_eval(constructor, X_trainval = (feature_input, raw_predictions), y_trainval = obs_input, generator = generator, fit_kwargs = fit_kwargs, return_predictions = True, scale_cv_mode = crossval_scaling)

generator.reset()

"""
Test RF Hybrid model only empirical info and dynamical info of the intermediate variables
"""
#sys.path.append(os.path.expanduser('~/Documents/Weave/'))
#from Weave.models import HybridExceedenceModel, fit_predict_evaluate
#
#forecasts = forc_trainval.copy().iloc[:,focus_class] # To include forecast probability too
#forecasts.name = 'pi'
#final_trainval = final_trainval.join(forecasts)
#
#params = dict(fit_base_to_all_cv = True, max_depth = 5, n_estimators = 2500, min_samples_split = 30, max_features = 1.0, n_jobs = 5)
#evaluate_kwds = dict(scores = [brier_score_loss], score_names = ['brier'])
#
#results3 = pd.DataFrame(np.nan, index = pd.MultiIndex.from_product([generator.groupids, ['train','val']], names = ['fold','part']), columns = ['brier'])
#for i, (trainind, valind) in enumerate(generator): # Entering the crossvalidation
#    h = HybridExceedenceModel(**params)
#    scores = fit_predict_evaluate(model = h, X_in = final_trainval.iloc[trainind,:], y_in = obs_trainval.iloc[trainind,focus_class],X_val = final_trainval.iloc[valind,:], y_val = obs_trainval.iloc[valind,focus_class], evaluate_kwds = evaluate_kwds)
#    results3.loc[(i,'train'),:] = scores.loc['brier']
#    results3.loc[(i,'val'),:] = scores.loc['brier'] * scores.loc['brier_val/train']
#
#generator.reset()

"""
Benchmarks
(Logistic) regression was fitted on all train/validation data, and to the single focus class
Revised predict method is called here
"""
if multi_eval:
    if hasattr(generator,'groupids'):
        benchmarks = pd.DataFrame(np.nan, index = pd.MultiIndex.from_product([generator.groupids, ['val'], ['raw','trend']], names = ['fold','part','reference']), columns = ['brier'])
    else:
        benchmarks = pd.DataFrame(np.nan, index = pd.MultiIndex.from_product([[0], ['val'], ['raw','trend']], names = ['fold','part','reference']), columns = ['brier'])
    for i, (trainind, valind) in enumerate(generator): # Entering the crossvalidation
        benchmarks.loc[(i,'val','raw'),:] = np.mean((obs_trainval_bool.iloc[valind] - forc_trainval.iloc[valind,focus_class])**2)
        benchmarks.loc[(i,'val','trend'),:] = np.mean((obs_trainval_bool.iloc[valind] - lr.predict(time_input)[valind])**2)
    
    bs_joined = pd.merge(results.loc[(slice(None),'val'),'brier'],benchmarks, left_index = True, right_index = True, suffixes = ['_pp','_ref'])
    bss = (1 - bs_joined['brier_pp']/bs_joined['brier_ref']).unstack('reference')

    #bss = 1 - results.loc[(slice(None),'val'),'brier'] / benchmarks.loc[(slice(None),'val','trend'),'brier']
    print(bss.round(3))
else: # RPSS benchmarks
    benchmarkraw = ranked_prob_score(forc_trainval.values, obs_trainval.values)
    benchmarktrend = ranked_prob_score(lr.predict_proba(time_input), obs_trainval.values)
    #print(f'RPS_raw    RPS_trend: ')
    #print(f'{np.round(benchmarkraw, 3)}       {np.round(benchmarktrend, 3)}')
    print(f'RPSS_raw    RPSS_trend: ')
    print(f'{np.round(1 - score / benchmarkraw, 3)}       {np.round(1 - score / benchmarktrend, 3)}')

generator.reset()

"""
Statistics in the cv 
"""
stats = {}
for i, (trainind, valind) in enumerate(generator): # Entering the crossvalidation
    stats.update({('obs',i,'train'):obs_trainval.iloc[trainind,focus_class].mean()})
    stats.update({('obs',i,'val'):obs_trainval.iloc[valind,focus_class].mean()})
    stats.update({('for',i,'train'):forc_trainval.iloc[trainind,focus_class].mean()})
    stats.update({('for',i,'val'):forc_trainval.iloc[valind,focus_class].mean()})
stats = pd.Series(stats)
#obs.groupby(obs.index.get_level_values('time').year).sum()
#forc.groupby(forc.index.get_level_values('time').year).mean()
#classes,groups = generate_balanced_kfold(forc.loc[:,1], shuffle = True)

"""
danger zone
"""
#if crossval_scaling:
#    feature_input, feature_scaler = scale_other_features(final_trainval) 
#model = constructor.fresh_model()
###fit_kwargs['shuffle'] = False
##fit_kwargs['epochs'] = 20
#model.fit(x = [feature_input, raw_predictions], y=obs_input, validation_split = 0.4, **fit_kwargs)
#
#time_test = time_scaler.transform(obs_test.index.get_level_values('time').to_julian_date()[:,np.newaxis])
#feature_test = feature_scaler.transform(X_test)
#raw_test = multiclass_log_forecastprob(forc_test)
#score = model.evaluate([feature_test,raw_test],obs_test.values)
#test_pred = model.predict([feature_test,raw_test])
#train_pred = model.predict([feature_input, raw_predictions]) 
#print(np.mean((obs_test.iloc[:,focus_class] - forc_test.iloc[:,focus_class])**2))
#print(np.mean((obs_test.iloc[:,focus_class] - lr.predict(time_test))**2))
