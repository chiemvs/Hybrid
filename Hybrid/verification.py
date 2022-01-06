"""
Mostly combination code to handle raw matched sets (from SubSeas)
To train and add fitted neural models.
And to add benchmark models.
"""
import os 
import sys
import warnings
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from pathlib import Path
from typing import Callable, Union, List

from .dataprep import test_trainval_split, default_prep, scale_time, binarize_hotday_predictand, singleclass_regression # We do not use base-exceedence from Weave.models because the scaler is hidden
from .neuralnet import DEFAULT_FIT

sys.path.append(os.path.expanduser('~/Documents/SubSeas'))
from observations import Climatology
from forecasts import ModelClimatology
from comparison import ForecastToObsAlignment, Comparison

sys.path.append(os.path.expanduser('~/Documents/Weave'))
from Weave.utils import max_pev

def load(booksname: str, compute = False):
    al = ForecastToObsAlignment('JJA','45r1')
    al.recollect(booksname = booksname)
    al.alignedobject['separation'] = al.alignedobject['leadtime'] - 1
    if compute:
        return(al.alignedobject.compute())
    else:
        return al

def add_trend_model(df: pd.DataFrame, groupers = ['leadtime','clustid'], exclude_test: bool = False, return_coefs = False):
    """
    Modifies the dataframe inplace, adding a trend column 
    This can involve multiple logistic regressions. namely one for a series with a unique time_index.
    Determined by groupers. Pooling (non-grouping) leadtime might be possible, 
    but still needs unique time for reindex
    """
    smalldf = df[['observation']]
    if smalldf.columns.nlevels == 2:
        smalldf.columns = smalldf.columns.droplevel(-1)
    groupeddf = smalldf.groupby(groupers) # Accomodating the fact that multiple regions might be included
    coefs = []
    intercepts = []
    predictions = []
    keys = []
    for key, subdf in groupeddf:
        if exclude_test:
            subdf_test, subdf_trainval, _ = test_trainval_split(subdf.sort_index(), crossval = True, nfolds = 3, balanced = True)
            print(f'excluding test years {subdf_test.index.get_level_values("time").year.unique().values} from logistic trend fit') 
        else:
            subdf_trainval = subdf
        # Fitting. Regression function extracts and normalizes the time index of the target 
        time_trainval_scaled, time_scaler, logreg = singleclass_regression(binary_obs = subdf_trainval['observation'])
        # Generation of predictions
        time_scaled, _ = scale_time(subdf, fitted_scaler=time_scaler)
        predictions.append(pd.Series(logreg.predict(time_scaled), index = subdf.index , name = 'trend'))
        coefs.append(logreg.coef_[0][0])
        intercepts.append(logreg.intercept_[0])
        keys.append(key)
    predictions = pd.concat(predictions, axis = 0).reindex(df.index) # Should be the same order now as the non-indexed df
    df['trend'] = predictions.values
    if return_coefs:
        logreg_results = pd.DataFrame({'coef':coefs, 'intercept':intercepts}, index = pd.MultiIndex.from_tuples(keys, names = groupers))
        return logreg_results

def _compute_score(df, scorename: str, scorefunc: Callable, args: tuple = tuple(), kwargs: dict = {}):
    """
    Computation of scores for present columns 
    in relation to the observation
    takes only those are not yet computed (no {column}_{scorename} present)
    scorefunc should accept (y_true, y_pred)
    """
    for key in ['pi','ppjm','ppsf','climatology','trend']:
        newkey = f'{key}_{scorename}'
        if (key in df.columns) and (not (newkey in df.columns)):
            score = scorefunc(df['observation'].values.squeeze(), df[key].values.squeeze(), *args, **kwargs)
            df.loc[:,newkey] = score
    return df

def compute_bs(df):
    def bs(y_true, y_pred):
        return (y_pred - y_true)**2
    return _compute_score(df, scorename = 'bs', scorefunc = bs)

def compute_bss(df):
    """
    Reduces the dataframe, currently does not compute for a potential 'pp' column
    carrying post-processed probabilities
    """
    bs_cols = df.columns.get_level_values(0).str.endswith('bs')
    mean_bs = df.iloc[:,bs_cols].groupby(['separation','clustid']).mean() # Mean over time
    if means_bs.columns.nlevels == 2:
        mean_bs.columns = mean_bs.columns.droplevel('number')
    for key in ['pi','ppsf','ppjm']:
        try:
            mean_bs[f'{key}_climatology_bss'] = 1 - mean_bs['{key}_bs'] / mean_bs['climatology_bs']
            mean_bs[f'{key}_trend_bss'] = 1 - mean_bs['{key}_bs'] / mean_bs['trend_bs']
        except KeyError:
            pass
    return mean_bs.iloc[:,mean_bs.columns.str.endswith('bss')]

def compute_kss(df):
    assert len(df.index.get_level_values('clustid').unique()) == 1, 'kuipers skill score aggregates the whole frame to a single float, so can only be called on a homogeneous set, no multiple clustids allowed'
    warnings.warn('kuipers skill score aggregates the whole frame to a single float, so can only be called on a homogeneous set. Check the leadtimes you are pooling')
    return _compute_score(df, scorename = 'kss', scorefunc = max_pev)

def compute_auc(df):
    assert len(df.index.get_level_values('clustid').unique()) == 1, 'AUC score aggregates the whole frame to a single float, so can only be called on a homogeneous set, no multiple clustids allowed'
    warnings.warn('AUC score aggregates the whole frame to a single float, so can only be called on a homogeneous set. Check the leadtimes you are pooling')
    return _compute_score(df, scorename = 'auc', scorefunc = roc_auc_score)

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
    df = df.set_index(['time','separation', 'clustid'])
    # Adding reference forecasts, namely the constant prediction, and potentially the trend
    if add_trend:
        coefs = add_trend_model(df = df, groupers = ['separation','clustid'], exclude_test = True, return_coefs=return_trend)
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
    df = df.set_index(['time','separation', 'clustid'])
    if add_trend:
        coefs = add_trend_model(df = df, groupers = ['separation','clustid'], exclude_test = True, return_coefs=return_trend) # happens inplace
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

def build_fit_nn_model(predictandname, add_trend: bool = True, npreds: int = None, use_jmeasure: bool = False, return_separate_test: bool = True):
    """
    Uses the default data preparation (from dataprep), then fits a default model
    """ 
    focus_class = -1
    prepared_data, constructor = default_prep(predictandname = predictandname, npreds = npreds, use_jmeasure = use_jmeasure, focus_class = focus_class)
    if use_jmeasure:
        signature = 'jm'
    else:
        signature = 'sf'
    # Some climatological information, could for tganom predictands also be derived from the name.
    # But not for tg-ex
    climprob = prepared_data.raw.obs.mean(axis = 0).iloc[-1]

    # Training the model
    model = constructor.fresh_model()
    model.fit(x = prepared_data.neural.trainval_inputs, y=prepared_data.neural.trainval_output, validation_split = 0.33, **DEFAULT_FIT)
    
    # Generating model predictions:
    preds_test = model.predict(x = prepared_data.neural.test_inputs)
    preds_trainval = model.predict(x = prepared_data.neural.trainval_inputs)

    total = pd.DataFrame({'pi': np.concatenate([prepared_data.crossval.forc_trainval, prepared_data.crossval.forc_test])[:,focus_class], 
            f'pp{signature}':np.concatenate([preds_trainval, preds_test])[:,focus_class],
            'climatology':climprob}, index = prepared_data.crossval.forc_trainval.index.union(prepared_data.crossval.forc_test.index, sort = False))

    total['observation'] = np.concatenate([prepared_data.crossval.obs_trainval, prepared_data.crossval.obs_test])[:,focus_class]
    # Add extra index level to match the levels of non-post-processed tganom or tgex sets
    total.index = pd.MultiIndex.from_frame(total.index.to_frame().assign(clustid = 9))

    # Training a trend benchmark model, and generating scale test time input
    if add_trend:
        coefs = add_trend_model(df = total, groupers = ['separation','clustid'], exclude_test = True, return_coefs=False) # happens inplace

    total = compute_bs(total)
    
    if return_separate_test:
        return total, total.iloc[total.index.droplevel(-1).get_indexer(prepared_data.crossval.obs_test.index),:].copy()
    else:
        return total

def reduce_to_ranks(df):
    """
    Reduce the whole frame to rankings (one per score)
    so make sure you supply the homogeneous set
    """
    df = df.copy()
    if df.columns.nlevels == 2:
        df.columns = df.columns.droplevel(-1)
    returns = []
    for scorename, ascending in zip(['bs','auc','kss'],[False,True,True]):
        cols = df.columns.str.endswith(scorename)
        if cols.any():
            ranks = df.iloc[:,cols].mean().rank(ascending = ascending) # Highest rank = best
            ranks.index = pd.MultiIndex.from_product([[scorename],[s.split('_')[0] for s in ranks.index]], names = ['score','forecast'])
            returns.append(ranks)
    return pd.concat(returns, axis = 0)

def reduce_to_skill(df):
    """
    Reduces the whole frame to skill scores
    (First average then normalize)
    The benchmark is the climatology
    Kuipers is already a skillscore basically (sperf = 1, sref_climatology = 0)
    """
    df = df.copy()
    if df.columns.nlevels == 2:
        df.columns = df.columns.droplevel(-1)
    returns = []
    for scorename, score_perf in zip(['bs','auc','kss'],[0,1,1]):
        cols = df.columns.str.endswith(scorename)
        if cols.any():
            meanscore = df.iloc[:,cols].mean()
            score_ref = meanscore.loc[f'climatology_{scorename}']
            skillscore = (score_ref - meanscore)/(score_ref - score_perf) 
            skillscore.index = pd.MultiIndex.from_product([[scorename],[s.split('_')[0] for s in skillscore.index]], names = ['score','forecast'])
            returns.append(skillscore)
    return pd.concat(returns, axis = 0)
