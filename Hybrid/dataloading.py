import sys
import os
import warnings
import pandas as pd
import numpy as np

from typing import Union, Callable, Tuple, List

from .dataprep import categorize_hotday_predictand, one_hot_encoding

sys.path.append(os.path.expanduser('~/Documents/Weave/'))
from Weave.models import map_foldindex_to_groupedorder
sys.path.append(os.path.expanduser('~/Documents/SubSeas/'))
from comparison import ForecastToObsAlignment, Comparison
from observations import Climatology
from forecasts import ModelClimatology

"""
Preparation of a combined (dynamic and empirical) input output set for any statistical model
including train validation test splitting
including scaling
Currently only time series
"""

def read_dynamic_data(booksname: str, separation: Union[int,slice,List[int]], clustids: Union[int,slice,List[int]]) -> pd.DataFrame:
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

def read_climate_index(filepath : str, separation: Union[int,slice,List[int]]) -> pd.DataFrame:
    """
    Reads pre-lagged daily RMM mjo index, originally from http://www.bom.gov.au/climate/mjo/graphics/rmm.74toRealtime.txt
    A two-component index (rmm1 and rmm2) (of which phase and amplitude are derivatives)
    Not possible to select a clustid because constant (always 1, a dummy value)

    Or reads the pdo index, supplied by Sem (also daily, and also a dummy cluster)
    """
    if isinstance(separation, int):
        separation = slice(separation,separation)
    df = pd.read_hdf(filepath) 
    if 'mjo' in df.columns.get_level_values('variable'):
        df = df.iloc[:,df.columns.get_level_values('metric').isin(['rmm1','rmm2'])] # Drops phase and amplitude
    return df.loc[(slice(None),separation),:]

def read_tganom_predictand(booksname: str, separation: Union[int,slice,List[int]], clustid: int, climname: str = None, modelclimname: str = None) -> Tuple[pd.Series,pd.Series]:
    """
    Functionality from SubSeas to get doy based climatology thresholds (for observation)
    and doy/leadtime based modelclimatology threshold (for forecast probability) involved in binarization
    outputs two-class versions, one frame for forecast, one for observation
    """
    al = ForecastToObsAlignment('JJA','45r1')
    al.recollect(booksname = booksname)
    al.alignedobject['separation'] = al.alignedobject['leadtime'] - 1
    cl = Climatology('tg-anom', name = climname)
    cl.localclim()
    if not modelclimname is None:
        mcl = ModelClimatology('45r1','tg', **{'name':modelclimname})
        mcl.local_clim()
    else:
        mcl = None
    comp = Comparison(al, climatology = cl, modelclimatology = mcl)
    comp.brierscore() # Transforms into exceedences, adds pi (tukey)
    frame = comp.frame.compute()
    frame = frame.set_index(['time','clustid','separation']).sort_index()
    subset = frame.loc[(slice(None),clustid,separation),:]
    subset.index = subset.index.droplevel('clustid')
    obs = pd.DataFrame(one_hot_encoding(subset[('observation',0)]), index = subset.index, columns = pd.Index([0,1], name = 'categoryid'))
    forecast = pd.DataFrame(np.concatenate([1 - subset[['pi']].values, subset[['pi']].values], axis = 1), index = subset.index, columns = pd.Index([0,1], name = 'categoryid')) 
    return obs, forecast 

def read_raw_predictand(booksname: str, clustid: int, separation: Union[int,slice,List[int]], dynamic_prediction_too: bool = False) -> Union[pd.Series,Tuple[pd.Series,pd.DataFrame]]:
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

def read_raw_predictor_ensmean(booksname: str, clustid: Union[int,slice,List[int]], separation: Union[int,slice,List[int]]) -> pd.DataFrame:
    """
    Currently a forecast from a 'matched' set for an 'intermediate variable' like block soil moisture
    Involves some reshaping of the match frame such that the columns multi-index will
    be ready for a merge with the empirical data 
    """
    df = read_dynamic_data(booksname = booksname, separation = separation, clustids = clustid)
    ensmean = df['forecast'].mean(axis = 1)
    return ensmean.unstack('clustid') # clustid into the columns (different predictors)

def read_raw_predictor_regimes(booksname: str, clustid: Union[int,slice,List[int]], separation: Union[int,slice,List[int]], observation_too: bool = False) -> pd.DataFrame:
    """
    Loading of the forecast from a matched regime set
    """
    df = read_dynamic_data(booksname = booksname, separation = separation, clustids = clustid)
    df.columns = df.columns.droplevel('number')
    if observation_too:
        return df['forecast'].unstack('clustid'), df['observation'].unstack('clustid')
    else:
        return df['forecast'].unstack('clustid')

def annotate_raw_predictor(predictor: pd.DataFrame, variable: str, timeagg: int, metric: str = 'mean') -> pd.DataFrame:
    """
    Expands the column index with levels for variable and timeagg
    To get: variable, timeagg, clustid, metric
    happens in place
    """
    assert (predictor.columns.nlevels == 1) and (predictor.columns.name == 'clustid'), 'Expecting one prior level, namely clustid'
    predictor.columns = pd.MultiIndex.from_product([[variable], [timeagg], predictor.columns, [metric]], names = ['variable','timeagg','clustid','metric']) 


def read_empirical_predictors(filepath: str, separation: Union[int,slice,List[int]], timeagg: Union[int,slice,List[int]] = slice(None)):
    """
    Bulk read of all empirical predictors (multiple variables and clusters)
    Will be filtered later but optional to already select a subset of timeaggs 
    """
    df = pd.read_parquet(filepath)
    # We must read fold 4, because this is the only one not using a large part of our data for training.
    # Only one with a good possibility of testing in 2013-2019
    map_foldindex_to_groupedorder(df, n_folds = 5)
    df.columns = df.columns.droplevel('lag') # redundant with separation in place. 
    old_separations = df.columns.levels[df.columns.names.index('separation')]# Here we handle positive separations instead of negative
    df.columns = df.columns.set_levels(-old_separations, level = 'separation')
    if isinstance(timeagg, int):
        timeagg = slice(timeagg,timeagg)
    if isinstance(separation, int):
        separation = [separation] #slice(separation,separation)
    df = df[4].loc[:,(slice(None),timeagg, separation)] # fold level will drop out
    return df.stack('separation')

def prepare_full_set(predictand_name, ndaythreshold: Union[List[int],int], predictand_cluster: int = 9 , leadtimepool: Union[List[int],int] = 15) -> Tuple[pd.DataFrame,pd.Series,pd.Series]:
    """
    Prepares predictors and predictand
    Currently uses very simple block-mean ensemble predictors
    leadtimepool is in days of absolute separation (1 means 1 full day between issuing of forecast and start of the event)
    returns (empirical + dynamical predictors, dynamical forecasts of predictand, observed predictand value)
    Multiple ndaythresholds leads to multiclass predictions
    """
    simple_dynamical_set = pd.DataFrame({'booksname':[
        'books_paper3-3-simple_swvl4-anom_JJA_45r1_21D-roll-mean_1-swvl-simple-mean.csv',
        'books_paper3-3-simple_swvl4-anom_JJA_45r1_31D-roll-mean_1-swvl-simple-mean.csv',
        'books_paper3-3-simple_swvl13-anom_JJA_45r1_21D-roll-mean_1-swvl-simple-mean.csv',
        'books_paper3-3-simple_swvl13-anom_JJA_45r1_31D-roll-mean_1-swvl-simple-mean.csv',
        'books_paper3-3-simple_z-anom_JJA_45r1_21D-roll-mean_1-swvl-simple-mean.csv',
        'books_paper3-3-simple_z-anom_JJA_45r1_31D-roll-mean_1-swvl-simple-mean.csv',
        'books_paper3-3-simple_sst-anom_JJA_45r1_21D-roll-mean_1-sst-simple-mean.csv',
        'books_paper3-3-simple_sst-anom_JJA_45r1_31D-roll-mean_1-sst-simple-mean.csv',
        'books_paper3-4-4regimes_z-anom_JJA_45r1_21D-frequency_ids.csv',
        ],
        'readfunc':[
        read_raw_predictor_ensmean,
        read_raw_predictor_ensmean,
        read_raw_predictor_ensmean,
        read_raw_predictor_ensmean,
        read_raw_predictor_ensmean,
        read_raw_predictor_ensmean,
        read_raw_predictor_ensmean,
        read_raw_predictor_ensmean,
        read_raw_predictor_regimes,
        ],
        'timeagg':[21,31,21,31,21,31,21,31,21],
        'metric':['mean','mean','mean','mean','mean','mean','mean','mean','freq'],
        }, index = pd.MultiIndex.from_tuples([('swvl4',21),('swvl4',31),('swvl13',21),('swvl13',31),('z',21),('z',31),('sst',21),('sst',31),('z-reg',21)]))

    dynamical_predictors = [] 
    for var, timeagg in simple_dynamical_set.index:
        readfunc = simple_dynamical_set.loc[(var,timeagg),'readfunc']
        predictor = readfunc(
                booksname = simple_dynamical_set.loc[(var,timeagg),'booksname'], 
                clustid = slice(None),
                separation = leadtimepool)
        annotate_raw_predictor(predictor, 
                variable = var, 
                timeagg = simple_dynamical_set.loc[(var,timeagg),'timeagg'],
                metric = simple_dynamical_set.loc[(var,timeagg),'metric'])
        dynamical_predictors.append(predictor)
    dynamical_predictors = pd.concat(dynamical_predictors, join = 'inner', axis = 1)

    # Now comes the selection of data
    empirical_available_at = [0,1,3,5,7,11,15,21,31] # leadtime in days
    if isinstance(leadtimepool, list):
        """
        In this case we're going to load a single lead time pattern which is projected to the others 
        As the clustids don't correspond over the leadtimes (they can come into existence, or actually switch location).
        """
        longest_shared_leadtime = max(leadtimepool)
        assert (longest_shared_leadtime in [15,21]), 'currently only 15 and 21 day lag patterns are projected to lower leadtimes. So pick 15 or 21 as the max in your leadtime pool'
        empiricalfile = f'/nobackup_1/users/straaten/clusters_cv_spearmanpar_varalpha_strict/precursor.multiagg.-{longest_shared_leadtime}.parquet'
        warnings.warn(f'picking values from pattern at {longest_shared_leadtime} which got projected onto the shorter leadtimes in {leadtimepool}') 
    else:
        assert leadtimepool in empirical_available_at, f'single chosen leadtime {leadtimepool} should be one of the empirically available: {empirical_available_at}'
        empiricalfile = '/nobackup_1/users/straaten/clusters_cv_spearmanpar_varalpha_strict/precursor.multiagg.parquet'

    empirical_set = read_empirical_predictors(empiricalfile, separation = leadtimepool) 
    empirical_dynamical_set = dynamical_predictors.join(empirical_set, how = 'left') # Empirical predictors cover more data

    # Adding climate indices, very long timeseries, so no outer or right join
    mjo = read_climate_index(filepath = '/nobackup/users/straaten/predsets/mjo_daily.h5', separation = leadtimepool)
    pdo = read_climate_index(filepath = '/nobackup/users/straaten/predsets/pdo_daily.h5', separation = leadtimepool)
    empirical_dynamical_set = empirical_dynamical_set.join(mjo, how = 'left').join(pdo, how = 'left')
    

    observations, forecast = read_raw_predictand(booksname = predictand_name, clustid = predictand_cluster, separation = leadtimepool, dynamic_prediction_too = True)
    if isinstance(ndaythreshold, int):
        print(f'Binarized target: n_hotdays >= {ndaythreshold}, onehot-encoded')
        ndaythreshold = [ndaythreshold]

    else:
        print(f'Multiclass target: n_hotdays >= {ndaythreshold}, onehot-encoded')
    observations, bounds = categorize_hotday_predictand(observations,  lower_split_bounds = ndaythreshold)
    predicted_predictand, bounds = categorize_hotday_predictand(forecast,  lower_split_bounds = ndaythreshold)
    observations = pd.DataFrame(one_hot_encoding(observations), index = observations.index, columns = bounds.index)


    # Getting the lengths equal
    index_intersection = empirical_dynamical_set.index.intersection(predicted_predictand.index).intersection(observations.index).sort_values()

    return empirical_dynamical_set.loc[index_intersection,:], predicted_predictand.loc[index_intersection], observations.loc[index_intersection]
    
