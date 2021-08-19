import sys
import os
import warnings
import pandas as pd
import numpy as np

from typing import Union

sys.path.append(os.path.expanduser('~/Documents/Weave/'))
from Weave.models import map_foldindex_to_groupedorder
sys.path.append(os.path.expanduser('~/Documents/SubSeas/'))
from comparison import ForecastToObsAlignment

"""
Preparation of a combined (dynamic and empirical) input output set for any statistical model
including train validation test splitting
Currently only time series

TODO: scaling should be here
TODO: pre-selection without a model (J-measure, pdf-overlap) should be here
"""

def read_dynamic_data(booksname: str, separation: Union[int,slice,list[int]], clustids: Union[int,slice,list[int]]) -> pd.DataFrame:
    """
    Reads 'matched' sets generated with SubSeas code
    Separation (days) is basically leadtime stamped at start of a period minus 1 
    Values should be positive and can be multiple
    All cluster ids or a single one are read
    """
    al = ForecastToObsAlignment('JJA','45r1')
    al.recollect(booksname = booksname)
    modelinfo = al.alignedobject.compute()
    modelinfo['separation'] = modelinfo['leadtime'] - 1
    modelinfo = modelinfo.set_index(['time','clustid','separation']).sort_index()
    if isinstance(clustids, int):
        clustids = slice(clustids,clustids) # Make sure that it does not drop from the index
    if isinstance(separation, int):
        separation = slice(separation,separation)
    subset = modelinfo.loc[(slice(None),clustids,separation),['forecast','observation']]
    return subset

def read_raw_predictand(booksname: str, clustid: int, separation: Union[int,slice,list[int]], dynamic_prediction_too: bool = False) -> Union[pd.Series,tuple[pd.Series,pd.DataFrame]]:
    """
    Currently the 'observation' from a 'matched' set.
    Also possible to read the 'forecast' from this set, then this is returned too
    Which can form a special input 
    Usually a specific target cluster of the predictand needs to be read
    """
    assert isinstance(clustid, int), 'Only one cluster can be selected as target'
    df = read_dynamic_data(booksname = booksname, separation = separation, clustids = clustid)
    # If we are pooling leadtimes then we will have duplicates in a time index 
    # Therefore we need the separation axis to do an eventual merge to predictors
    df.index = df.index.droplevel('clustid')
    observation = df['observation'].iloc[:,0]
    observation.name = 'observation'
    if dynamic_prediction_too:
        return observation, df['forecast']
    else:
        return observation 

def read_raw_predictor_ensmean(booksname: str, clustid: Union[int,slice,list[int]], separation: Union[int,slice,list[int]]) -> pd.DataFrame:
    """
    Currently a forecast from a 'matched' set for an 'intermediate variable' like block soil moisture
    Involves some reshaping of the match frame such that the columns multi-index will
    be ready for a merge with the empirical data 
    """
    df = read_dynamic_data(booksname = booksname, separation = separation, clustids = clustid)
    ensmean = df['forecast'].mean(axis = 1)
    return ensmean.unstack('clustid') # clustid into the columns (different predictors)

def annotate_raw_predictor(predictor: pd.DataFrame, variable: str, timeagg: int, metric: str = 'mean') -> pd.DataFrame:
    """
    Expands the column index with levels for variable and timeagg
    To get: variable, timeagg, clustid, metric
    happens in place
    """
    assert (predictor.columns.nlevels == 1) and (predictor.columns.name == 'clustid'), 'Expecting one prior level, namely clustid'
    predictor.columns = pd.MultiIndex.from_product([[variable], [timeagg], predictor.columns, [metric]], names = ['variable','timeagg','clustid','metric']) 


def read_empirical_predictors(filepath: str, separation: int, timeagg: Union[int,slice,list[int]] = slice(None)):
    """
    Bulk read of all empirical predictors (multiple variables and clusters)
    Will be filtered later but optional to already select a subset of timeaggs 
    """
    assert isinstance(separation, int) and (separation in [0,1,3,5,7,11,15,21,31]), 'Only a single separation can be loaded, as clustids are discontinuous over leadtimes, i.e. their locations differ.'
    df = pd.read_parquet(filepath)
    # We must read fold 4, because this is the only one not using a large part of our data for training.
    # Only one with a good possibility of testing in 2013-2019
    map_foldindex_to_groupedorder(df, n_folds = 5)
    df.columns = df.columns.droplevel('lag') # redundant with separation in place. 
    old_separations = df.columns.levels[df.columns.names.index('separation')]# Here we handle positive separations instead of negative
    df.columns = df.columns.set_levels(-old_separations, level = 'separation')
    if isinstance(timeagg, int):
        timeagg = slice(timeagg,timeagg)
    df = df[4].loc[:,(slice(None),timeagg, separation)] # fold level will drop out
    return df.stack('separation')

def binarize_hotday_predictand(df: Union[pd.Series, pd.DataFrame], ndaythreshold: int) -> pd.Series:
    """
    Currently assumes it is the hotdays predictand
    For observations (pd.Series) it computes greater or equal than the ndaythreshold
    for forecasts it computes the probability (frequency of members) with greater or equal than the threshold 
    """
    if isinstance(df, pd.Series):
        return df >= ndaythreshold
    elif isinstance(df, pd.DataFrame):
        greater_equal =  df.values >= ndaythreshold # 2D array (samples, members)
        probability = greater_equal.sum(axis = 1) / float(df.shape[-1])
        return pd.Series(probability, index = df.index)
    else:
        raise ValueError('Wrong type of input')

def prepare_full_set(predictand_name, leadtimepool: Union[list[int],int] = 15) -> tuple[pd.DataFrame,pd.Series,pd.Series]:
    """
    Prepares predictors and predictand
    Currently uses very simple block-mean ensemble predictors
    leadtimepool is in days of absolute separation (1 means 1 full day between issuing of forecast and start of the event)
    returns (empirical + dynamical predictors, dynamical forecasts of predictand, observed predictand value)
    """
    empirical_available_at = [0,1,3,5,7,11,15,21,31] # leadtime in days
    if isinstance(leadtimepool, list):
        warnings.warn(f'Choosing a pool of leadtimes will duplicate empirical predictor values of a single from the available leadtimes {empirical_available_at}')

    simple_dynamical_set = pd.DataFrame({'booksname':[
        'books_paper3-3-simple_swvl4-anom_JJA_45r1_7D-roll-mean_1-swvl-simple-mean.csv',
        'books_paper3-3-simple_swvl13-anom_JJA_45r1_7D-roll-mean_1-swvl-simple-mean.csv',
        'books_paper3-3-simple_z-anom_JJA_45r1_7D-roll-mean_1-swvl-simple-mean.csv',
        'books_paper3-3-simple_sst-anom_JJA_45r1_7D-roll-mean_1-sst-simple-mean.csv',
        ],
        'timeagg':[7,7,7,7],
        'metric':['mean','mean','mean','mean'],
        }, index = ['swvl4','swvl13','z','sst'])

    # TODO fill this
    return empirical_dynamical_set, predicted_predictand, observations
    

"""
For the j-measure we expect some intrinsic discriminatory power when the binary predictand is trended and the inputs too. (Distribution conditional on positive will differ from the one conditional on negative)
"""

if __name__ == '__main__':
    leadtimepool = [4,5,6,7,8] 
    #name = 'books_paper3-2_tg-ex-q0.75-7D_JJA_45r1_1D_15-t2m-q095-adapted-mean.csv'
    #obs, forc = read_raw_predictand(name, 9, leadtimepool, True)
    name = 'books_paper3-3-simple_swvl4-anom_JJA_45r1_7D-roll-mean_1-swvl-simple-mean.csv'
    predictor = read_raw_predictor_ensmean(name, slice(None), leadtimepool) 
    annotate_raw_predictor(predictor, 'swvl4',7)

    file = '/nobackup_1/users/straaten/clusters_cv_spearmanpar_varalpha_strict/precursor.multiagg.parquet'
    """
    Bit of a problem that we don't have the empirical predictors at all leadtimes.
    Actually, the clustids don't correspond over the leadtimes (they can come into existence, or actually switch location).
    So better load only one of the available and if we want to pool leadtimes for empirical then fill with persistent values (the chosen leadtime value for the date that dynamically is predicted with another leadtime in the leadtimepool)
    """
    df = read_empirical_predictors(file, separation = 5) # 


