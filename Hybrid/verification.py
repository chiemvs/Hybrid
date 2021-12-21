"""
Mostly combination code to handle raw matched sets (from SubSeas)
To train and add fitted neural models.
And to add benchmark models.
"""
import os 
import sys
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from pathlib import Path

from .dataprep import binarize_hotday_predictand

sys.path.append(os.path.expanduser('~/Documents/Weave'))
from Weave.models import BaseExceedenceModel

sys.path.append(os.path.expanduser('~/Documents/SubSeas'))
from observations import Climatology
from forecasts import ModelClimatology
from comparison import ForecastToObsAlignment, Comparison

def load(booksname: str, compute = False):
    al = ForecastToObsAlignment('JJA','45r1')
    al.recollect(booksname = booksname)
    if compute:
        return(al.alignedobject.compute())
    else:
        return al

def add_trend_model(df, groupers = ['leadtime','clustid'], return_coefs = True):
    """
    Modifies the dataframe inplace, adding a trend column 
    This can involve multiple logistic regressions. namely one for a series with a unique time_index.
    Determined by groupers. Pooling (non-grouping) leadtime might be possible, 
    but still needs unique time for reindex
    """
    groupeddf = df.groupby(groupers)
    coefs = []
    intercepts = []
    predictions = []
    keys = []
    for key, subdf in groupeddf:
        model = BaseExceedenceModel()
        model.fit(X = subdf, y = subdf[('observation',0)]) # Extracts and normalizes the timeindex from X
        predictions.append(pd.Series(model.predict(X = subdf), index = subdf.index , name = 'trend'))
        coefs.append(model.coef_[0][0])
        intercepts.append(model.intercept_[0])
        keys.append(key)
    predictions = pd.concat(predictions, axis = 0).reindex(df.index) # Should be the same order now as the non-indexed df
    df['trend'] = predictions.values
    if return_coefs:
        logreg_results = pd.DataFrame({'coef':coefs, 'intercept':intercepts}, index = pd.MultiIndex.from_tuples(keys, names = groupers))
        return logreg_results

def compute_bs(df):
    """
    Computation of the BS values for those that are not yet computed
    """
    for key in ['pi','pp','climatology','trend']:
        newkey = f'{key}_bs'
        if (key in df.columns) and (not (newkey in df.columns)):
            bs = (df[key].values.squeeze() - df['observation'].values.squeeze())**2
            df[newkey] = bs
    return df

def compute_bss(df):
    """
    Reduces the dataframe, currently does not compute for a potential 'pp' column
    carrying post-processed probabilities
    """
    bs_cols = df.columns.get_level_values(0).str.endswith('bs')
    mean_bs = df.iloc[:,bs_cols].groupby(['leadtime','clustid']).mean() # Mean over time
    mean_bs.columns = mean_bs.columns.droplevel('number')
    mean_bs['bss_climatology'] = 1 - mean_bs['pi_bs'] / mean_bs['climatology_bs']
    if 'trend_bs' in mean_bs.columns:
        mean_bs['bss_trend'] = 1 - mean_bs['pi_bs'] / mean_bs['trend_bs']
        return mean_bs[['bss_climatology','bss_trend']]
    else:
        return mean_bs[['bss_climatology']]

def load_tganom_and_compute(bookfile: str, climname: str, modelclim: str = None, add_trend: bool = True, return_trend: bool = False):
    al = load(booksname = bookfile, compute = False)
    cl = Climatology('tg-anom', name = climname)
    cl.localclim() # Loads the precomputed climatology
    if not modelclim is None:
        mcl = ModelClimatology('45r1','tg', **{'name':modelclim})
        mcl.local_clim() # Loads the precomputed modelclimatology
    else:
        mcl = None
    comp = Comparison(al, climatology = cl, modelclimatology = mcl)
    comp.brierscore() # Transforms the observations in exceedences. Needed for trend model. 
    df = comp.frame.compute()
    df = df.set_index(['time','leadtime','clustid'])
    # Adding reference forecasts, namely the constant prediction, and potentially the trend
    if add_trend:
        coefs = add_trend_model(df = df, groupers = ['leadtime','clustid'], return_coefs=return_trend)
        df = compute_bs(df) # Also BS for the trend
    if return_trend:
        return df, coefs
    else:
        return df

def load_tgex_and_compute(bookfile: str, nday_threshold: int = 3, add_trend: bool = True, return_clim_freq: bool = False, return_trend: bool = False):
    """
    A computed aligned frame. returns it with a pi, pi_bs and clim_bs column. 
    Pi is for positive case so chance of >= nday_threshold
    possible to return the fitted logistic coefficients
    possible to return frequencies (these are namely not constant over clustids like with tganom > quantile)
    """
    df = load(booksname = bookfile, compute = True)
    df['pi'] = binarize_hotday_predictand(df['forecast'], ndaythreshold = nday_threshold) # Uses Tukey
    df['observation'] = binarize_hotday_predictand(df[('observation',0)], ndaythreshold = nday_threshold)
    clim_chances = df.groupby('clustid').mean()['observation'] # Assume there is little leadtime dependence (pooling these for a smoother estimate)
    clim_chances.columns = pd.MultiIndex.from_tuples([('climatology','')])
    df = df.merge(clim_chances, on = 'clustid')
    df = df.set_index(['time','leadtime','clustid'])
    if add_trend:
        coefs = add_trend_model(df = df, groupers = ['leadtime','clustid'], return_coefs=return_trend) # happens inplace
    df = compute_bs(df)
    returns = (df,)
    if return_clim_freq:
        returns = returns + (clim_chances,)
    if return_trend: # potentially already a tuple
        returns = returns + (coefs,)
    if len(returns) == 1:
        return returns[0]
    else:
        return returns

def load_compute_rank(bookfile: str, return_bias: bool = False):
    """
    Continous evaluation. Should not matter whether it is somewhat discrete, like the count of hotday exceedences.
    Actually multiple rank hists are possible, namely per leadtime and per clustid.
    Therefore returns an indexed dataframe. From which the desired 
    Option to return the bias between ensemble mean and observation
    """
    frame = load(booksname = bookfile, compute = True)
    n_members = frame['forecast'].shape[-1]
    noisy_for = frame['forecast'].values + np.random.normal(scale = 0.001, size = frame['forecast'].shape)
    noisy_obs = frame['observation'].values + np.random.normal(scale = 0.001, size = frame['observation'].shape)
    n_higher = (noisy_for > noisy_obs).sum(axis = 1)
    frame['placement'] = (-n_higher) + n_members + 1 # placement in the order. if 0 higher than place n+1,
    bin_edges = np.arange(1 - 0.5, n_members + 1 + 1 + 0.5) # from min (1) to max (12) + 1, both +- 0.5. Max is plus 2 because of np.arange non-inclusive stops
    frame = frame.set_index(['leadtime','clustid','time'])
    if 'number' in frame.columns.names:
        frame.columns = frame.columns.droplevel('number')
    if not return_bias:
        return frame['placement'], bin_edges
    else:
        frame.loc[:,'bias'] = frame['forecast'].mean(axis = 1) - frame['observation'] #bias between ensmean - obs 
        return frame[['placement','bias']], bin_edges

def build_fit_nn_model(predictandname, add_trend: bool = True, npreds: int = None, return_separate_test: bool = True):
    """
    Uses the standard settings
    Like a predetermined objective set (either jmeasure or sequential forward if npreds is given )
    """
    from Hybrid.neuralnet import construct_modeldev_model, earlystop, BrierScore, ConstructorAndCompiler
    from Hybrid.dataprep import test_trainval_split, multiclass_log_forecastprob, singleclass_regression, multiclass_logistic_regression_coefficients, scale_other_features, scale_time

    for_obs_dir = Path('/nobackup/users/straaten/predsets/full/') 
    if npreds is None: # jmeasure, still objective
        predictor_dir = Path('/nobackup/users/straaten/predsets/jmeasure/')
        predictor_name = f'{predictandname}_jmeasure-dyn_predictors.h5'
    else: # sequential forward
        predictor_dir = Path('/nobackup/users/straaten/predsets/objective_balanced_cv/')
        predictor_name = f'{predictandname}_multi_d20_b3_predictors.h5'
    predictors = pd.read_hdf(predictor_dir / predictor_name, key = 'input').iloc[:,slice(npreds)]
    forc = pd.read_hdf(for_obs_dir / f'{predictandname}_forc.h5', key = 'input')
    obs= pd.read_hdf(for_obs_dir / f'{predictandname}_obs.h5', key = 'target')
    
    focus_class = -1
    # Splitting the sets.
    X_test, X_trainval, generator = test_trainval_split(predictors, crossval = True, nfolds = 3, balanced = True)
    forc_test, forc_trainval, generator = test_trainval_split(forc, crossval = True, nfolds = 3, balanced = True)
    obs_test, obs_trainval, generator = test_trainval_split(obs, crossval = True, nfolds = 3, balanced = True)

    # Prepare trainval data for the neural network
    features_trainval, feature_scaler = scale_other_features(X_trainval)
    logforc_trainval = multiclass_log_forecastprob(forc_trainval)
    obsinp_trainval = obs_trainval.values

    # Preparing the test set
    features_test, _ = scale_other_features(X_test, fitted_scaler=feature_scaler)
    logforc_test = multiclass_log_forecastprob(forc_test)

    # Setting up the model TODO: try to place these default_kwargs in Hybrid.neuralnet
    construct_kwargs = dict(n_classes = obs_trainval.shape[-1],
            n_hidden_layers= 1,
            n_features = features_trainval.shape[-1],
            n_hiddenlayer_nodes = 4)

    compile_kwargs = dict(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0014),
            metrics = ['accuracy',BrierScore(class_index = focus_class)])

    constructor = ConstructorAndCompiler(construct_modeldev_model, construct_kwargs, compile_kwargs)

    fit_kwargs = dict(batch_size = 32, epochs = 200, shuffle = True, 
            callbacks = [earlystop(patience = 7, monitor = 'val_loss')])

    # Training the model
    model = constructor.fresh_model()
    model.fit(x = [features_trainval, logforc_trainval], y=obsinp_trainval, validation_split = 0.33, **fit_kwargs)
    
    # Generating model predictions:
    preds_test = model.predict(x = [features_test, logforc_test])
    preds_trainval = model.predict(x = [features_trainval, logforc_trainval])

    #total = pd.DataFrame{'pi': forc.iloc[:,focus_class], 'pp':np.concatenate([preds

    # Training a trend benchmark model, and generating scale test time input
    if add_trend:
        time_input, time_scaler, lr = singleclass_regression(obs_trainval.iloc[:,focus_class], regressor = LogisticRegression)
        time_test, _ = scale_time(obs_test, fitted_scaler=time_scaler)

        # Generating benchmark predictions:
        trend_test = lr.predict(time_test)
        trend_trainval = lr.predict(time_input)

    return forc_test, forc_trainval, obs_test, obs_trainval, preds_test, preds_trainval 
