import sys
import os
import warnings
import pandas as pd
import numpy as np
import xarray as xr
try:
    import tensorflow as tf
except ImportError:
    pass

from typing import Union, Callable, Tuple, List
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression

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

def read_raw_predictand(booksname: str, clustid: int, separation: Union[int,slice,list[int]], dynamic_prediction_too: bool = False) -> Union[pd.Series,Tuple[pd.Series,pd.DataFrame]]:
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


def read_empirical_predictors(filepath: str, separation: int, timeagg: Union[int,slice,List[int]] = slice(None)):
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

def categorize_hotday_predictand(df: Union[pd.Series, pd.DataFrame], lower_split_bounds: List[int]) -> Tuple[pd.Series, pd.DataFrame]:
    """
    Lower bounds in days. E.g. [4,8,14] leads to four groups
    0>=x<4,4>=x<8,8>=x<13,14>=x<=x.max()
    Returns a series with the number of the category it falls in
    and a DataFrame of bounds
    Note that the Tukey plotting position is does properly sum up to 1 for two categories.
    To avoid probabilities of 0 (problems with logarithm) and simultaneously also of 1,
    we do have the option to do (m - a) / (M + 1 - 2a)
    where m is the rank of the observation in that set (positions 1 to 12)
    and where M is the number of positions (12) which is n_members + 1
    We choose to do Tukey plotting position (a = 1/3)
    """
    lower_split_bounds = np.array([0] + lower_split_bounds)
    lower_split_bounds.sort()
    bounds = pd.DataFrame({'low_inclusive':lower_split_bounds,'up_exclusive':lower_split_bounds[1:].tolist() + [np.inf]}, index = pd.RangeIndex(len(lower_split_bounds), name = 'categoryid'))

    vals = np.digitize(df, lower_split_bounds, right = False) - 1 # -1 such that indices equal categoryids
    if isinstance(df, pd.Series):
        return pd.Series(vals, index = df.index), bounds
    elif isinstance(df, pd.DataFrame):
        returns = pd.DataFrame(np.nan, index = df.index, columns = bounds.index)
        alpha = 1/3 
        n_positions = df.shape[-1] + 1
        for categoryid in bounds.index:
            n_present = (vals == categoryid).sum(axis = 1)
            rank = n_present + 1
            probability = (rank - alpha) / float(n_positions + 1 - 2*alpha)
            returns.loc[:,categoryid] = probability
        return returns, bounds
    else:
        raise ValueError('Wrong type of input')

def binarize_hotday_predictand(df: Union[pd.Series, pd.DataFrame], ndaythreshold: int) -> pd.Series:
    """
    For observations (pd.Series) it computes greater or equal than the ndaythreshold
    for forecasts it computes the probability (frequency of members) with greater or equal than the threshold (Tukey)
    """
    classprob, bounds = categorize_hotday_predictand(df, lower_split_bounds = [ndaythreshold]) # LAst class will be the desired one
    if isinstance(df, pd.Series):
        return classprob == bounds.index[-1] 
    elif isinstance(df, pd.DataFrame):
        return classprob.iloc[:,-1]
    else:
        raise ValueError('Wrong type of input')

def prepare_full_set(predictand_name, ndaythreshold: Union[list[int],int], predictand_cluster: int = 9 , leadtimepool: Union[list[int],int] = 15) -> tuple[pd.DataFrame,pd.Series,pd.Series]:
    """
    Prepares predictors and predictand
    Currently uses very simple block-mean ensemble predictors
    leadtimepool is in days of absolute separation (1 means 1 full day between issuing of forecast and start of the event)
    returns (empirical + dynamical predictors, dynamical forecasts of predictand, observed predictand value)
    Multiple ndaythresholds leads to multiclass predictions
    """
    simple_dynamical_set = pd.DataFrame({'booksname':[
        'books_paper3-3-simple_swvl4-anom_JJA_45r1_21D-roll-mean_1-swvl-simple-mean.csv',
        'books_paper3-3-simple_swvl13-anom_JJA_45r1_14D-roll-mean_1-swvl-simple-mean.csv',
        'books_paper3-3-simple_z-anom_JJA_45r1_21D-roll-mean_1-swvl-simple-mean.csv',
        'books_paper3-3-simple_sst-anom_JJA_45r1_21D-roll-mean_1-sst-simple-mean.csv',
        'books_paper3-4-regimes_z-anom_JJA_45r1_21D-frequency_ids.csv',
        ],
        'readfunc':[
        read_raw_predictor_ensmean,
        read_raw_predictor_ensmean,
        read_raw_predictor_ensmean,
        read_raw_predictor_ensmean,
        read_raw_predictor_regimes,
        ],
        'timeagg':[21,14,21,21,21],
        'metric':['mean','mean','mean','mean','freq'],
        }, index = ['swvl4','swvl13','z','sst','z-reg'])

    dynamical_predictors = [] 
    for var in simple_dynamical_set.index:
        readfunc = simple_dynamical_set.loc[var,'readfunc']
        predictor = readfunc(
                booksname = simple_dynamical_set.loc[var,'booksname'], 
                clustid = slice(None),
                separation = leadtimepool)
        annotate_raw_predictor(predictor, 
                variable = var, 
                timeagg = simple_dynamical_set.loc[var,'timeagg'],
                metric = simple_dynamical_set.loc[var,'metric'])
        dynamical_predictors.append(predictor)
    dynamical_predictors = pd.concat(dynamical_predictors, join = 'inner', axis = 1)

    # Now comes the selection and potential duplication of empirical data
    empiricalfile = '/nobackup_1/users/straaten/clusters_cv_spearmanpar_varalpha_strict/precursor.multiagg.parquet'
    empirical_available_at = [0,1,3,5,7,11,15,21,31] # leadtime in days
    if isinstance(leadtimepool, list):
        warnings.warn(f'Choosing a pool of leadtimes will duplicate empirical predictor values of a single from the available leadtimes {empirical_available_at}')
        longest_shared_leadtime = max(set(empirical_available_at).intersection(leadtimepool))
        warnings.warn(f'values from {longest_shared_leadtime} will be projected onto others in {leadtimepool}') 
    else:
        assert leadtimepool in empirical_available_at, f'single chosen leadtime {leadtimepool} should be one of the empirically available: {empirical_available_at}'
        longest_shared_leadtime = leadtimepool


    empirical_set = read_empirical_predictors(empiricalfile, separation = longest_shared_leadtime) 
    empirical_set.index = empirical_set.index.droplevel('separation') # A single separation is read so we can drop that and merge to dynamical (could mean suplication)
    empirical_dynamical_set = dynamical_predictors.join(empirical_set, on = 'time', how = 'inner')

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
    

class GroupedGenerator(object):
    """
    Yields the boolean index for where group != groupid and the index for where group == groupid
    Can be seen as yielding train indices first (largest portion) and validation indeces second (smallest portion)
    Stops when the unique groupids are exhausted.
    """
    def __init__(self, groups: np.ndarray):
        assert groups.ndim == 1, 'only an array of rank 1 is allowed to generate indices in one dimension'
        self.groupids = np.unique(groups)
        self.ngroups = len(self.groupids)
        self.groups = groups 
        self.ncalls = 0
    def __repr__(self):
        return f'GroupedGenerator for {self.ngroups}'
    def __iter__(self):
        return self
    def __next__(self) -> tuple[np.ndarray, np.ndarray]:
        if self.ncalls < self.ngroups:
            groupid = self.groupids[self.ncalls]
            wheretrue = self.groups == groupid 
            self.ncalls += 1
            return ~wheretrue, wheretrue # train, val
        raise StopIteration()
    def reset(self):
        self.ncalls = 0

class SingleGenerator(object):
    """ Yields only once the true indices that it has been supplied with and its inverse """
    def __init__(self, whereval: np.ndarray):
        assert (whereval.ndim == 1) and (whereval.dtype == bool), 'only a 1D array with boolean values can be used to yield those as indices'
        self.whereval = whereval
        self.ncalls = 0
    def __iter__(self):
        return self
    def __next__(self) -> tuple[np.ndarray, np.ndarray]:
        if self.ncalls < 1:
            self.ncalls += 1
            return ~self.whereval, self.whereval
        raise StopIteration()
    def reset(self):
        self.ncalls = 0

def test_trainval_split(df: Union[pd.Series, pd.DataFrame], crossval: bool = False, nfolds: int = 4) -> tuple[Union[pd.Series, pd.DataFrame],Union[GroupedGenerator,SingleGenerator]]:
    """
    Hardcoded train/val test split, supplied dataset should be indexed by time
    Returns a single test set and a single combined train/validation set,
    plus a generator that can be called for the right indices for the latter 
    crossval = True with nfolds = 4 will lead to 4 times different indices, which are split by year  
    """
    assert df.index.names[0] == 'time', 'Dataframe or series should be indexed by time at zeroth level'
    extra_levels_indexer = (slice(None),) * (df.index.nlevels - 1) # levels in addition to time

    test_time = slice('2017-01-01',None,None) # all the way till last available date
    trainval_time = slice(None,'2016-12-31',None)

    if isinstance(df, pd.DataFrame):
        testset = df.loc[(test_time,) + extra_levels_indexer,:]
        trainvalset = df.loc[(trainval_time,) + extra_levels_indexer,:]
    else:
        testset = df.loc[(test_time,) + extra_levels_indexer]
        trainvalset = df.loc[(trainval_time,) + extra_levels_indexer]

    if not crossval:
        val_time = slice('2013-01-01','2016-12-31',None)
        generator = SingleGenerator(trainvalset.index.get_loc_level(val_time,'time')[0]) # This passes the boolean array to the generator
    else:
        years = trainvalset.index.get_level_values('time').year
        unique_years = years.unique()
        assert nfolds <= len(unique_years), 'More folds than unique years requested. Unable to split_on_year.'
        groupsize = int(np.ceil(len(unique_years) / nfolds)) # Maximum even groupsize, except for the last fold, if unevenly divisible then last fold gets only the remainder.
        groups = (years - years.min()).map(lambda year: year // groupsize)  
        assert len(groups.unique()) == nfolds
        generator = GroupedGenerator(groups = groups.values)

    return testset, trainvalset, generator

def j_measure(fraction_positive: np.ndarray, fraction_negative: np.ndarray) -> float:
    """
    Basically the diversion between two binned distributions (conditioned)
    Based on KL-divergence. Code inspired by: https://github.com/thunderhoser/ai2es_xai_course/blob/fd5bf910e13f5596ab452ea6b79c540fde1e8fc2/ai2es_xai_course/utils/utils.py#L3044
    Compares binned frequencies, lengths should overlap and no zeros should be present 
    """
    components = (fraction_negative - fraction_positive)*np.log2(fraction_negative/fraction_positive)
    return components.sum()

def perkins(fraction_positive: np.ndarray, fraction_negative: np.ndarray) -> float:
    """
    A simple measure of divergence between two binned distributions.
    Counting the overlap basically
    Based on Perkins et al (2007) "Evaluation of the AR4 Climate Models’ Simulated Daily Maximum Temperature, Minimum Temperature, and Precipitation over Australia Using Probability Density Functions" 
    https://journals.ametsoc.org/view/journals/clim/20/17/jcli4253.1.xml
    sum_over_bins(min(fraction_in_one, fraction_in_other)), so ranges from 0 to 1
    """
    joined = np.concatenate([fraction_positive[:,np.newaxis], fraction_negative[:,np.newaxis]], axis = 1)
    return joined.min(axis = 1).sum()

def filter_predictor_set(predset: pd.DataFrame, observation: pd.Series, how: Callable = j_measure, nmost_important: int = 20, nbins: int = 20, min_samples_per_bin: int = 10, most_extreme_quantile: float = 0.95, return_measures: bool = False) -> pd.DataFrame:
    """
    Reduces the amount of features (columns) in the predictor set based on distriminatory power
    to the nmost_important for a binary predictand (assuming independence and only direct effect)
    compares (binned) distributions x|y=1 and x|y=0
    If these have little overlap then good discriminatory power
    for binary observations the options are 'j_measure' and 'perkins'
    Binning takes place with nbins from 1-extreme_quantile till extreme_quantile 
    """
    assert observation.dtype == bool, 'only binary predictand implemented'
    most_extreme_quantile = max(1 - most_extreme_quantile, most_extreme_quantile) # to make sure that we have monotonically increasing bins later
    X_y1 = predset.iloc[observation.values,:]
    X_y0 = predset.iloc[~observation.values,:]
    measures = pd.Series(np.nan, index= predset.columns, name = how.__name__)
    for key in predset.columns:
        bin_range = predset[key].quantile((1-most_extreme_quantile, most_extreme_quantile))
        bin_edges = np.linspace(*bin_range.values, num = nbins - 1) # actually the outside will also form bins
        hist_y1 = np.digitize(X_y1[key], bin_edges) # index == 0: <= bin_edges[0] 
        hist_y0 = np.digitize(X_y0[key], bin_edges) # index == nbins: > bin_edges[-1]
        # Start adding counts to a higher bin when there are too little observations
        for k in range(nbins):
            if (hist_y1 == k).sum() < min_samples_per_bin or (hist_y0 == k).sum() < min_samples_per_bin:
                # We have too little in this bin in either y1 or y0, or both
                print('too little in',k)
                if k < (nbins - 1):
                    newval = k+1 # merge into one cluster higher
                else: # if the last cluste we cannot shift upwards, so shift downwards to last maximum
                    newval = hist_y1[np.where(hist_y1 != k)].max()
                hist_y1[np.where(hist_y1 == k)] = newval
                hist_y0[np.where(hist_y0 == k)] = newval
        fraction_y1 = np.unique(hist_y1, return_counts = True)[-1] / len(hist_y1) # lengths should be equivalent
        fraction_y0 = np.unique(hist_y0, return_counts = True)[-1] / len(hist_y0)
        measures.loc[key] = how(fraction_y1, fraction_y0)

    ranked = measures.rank(method = 'first') # 1 to n (lowest to highest value)
    if how == j_measure: # Higher means more discriminatory power
        important = ranked > (len(ranked) - nmost_important)
    else: # perkins: lower (less overlap) = more dcriminatory power
        important = ranked <= nmost_important 
    keys = important.iloc[important.values].index
    if return_measures:
        return predset.loc[:,keys], measures
    else:
        return predset.loc[:,keys]
    
"""
Functions for prepping data for the specific neural nets 
"""

def one_hot_encoding(categorical_obs: Union[pd.Series, np.ndarray]) -> np.ndarray:
    """
    In as a 1D (categorical, probably binary) timeseries, out as a onehot encoded
    2D array (nsamples, nclasses)
    """
    classes = np.unique(categorical_obs)
    nclasses = len(classes)
    if nclasses == 1:
        warnings.warn(f'Only one class {classes} found for the one_hot encoding. Can be due to small sample size, otherwise check the data')
    return tf.one_hot(categorical_obs, depth = max(2,nclasses)).numpy()

def multiclass_log_forecastprob(probs: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
    """
    Generates logarithms of probabilities for the present classes
    These are required input for the Neural network 
    predicting deviations from the raw dynamic forecast probability.
    Returns array (nsamples, [logprob_firstclass,..., logprob_lastclass])
    So positive class last
    """
    assert (len(probs.shape) == 2) and (probs.shape[-1] > 1), 'probabilities of multiple classes should already be present'
    if isinstance(probs, pd.DataFrame):
        probs = probs.values
    return np.log(probs) 

def scale_time(df: Union[pd.Series,pd.DataFrame], fitted_scaler: MinMaxScaler = None) -> tuple[np.ndarray,MinMaxScaler]:
    """
    Preparation of time input array (n_samples, 1) 
    for logistic regression or a neural network leveraging logistic regression
    possible to supply a pre-fitted scaler (e.g. to transform test data)
    """
    time_input = df.index.get_level_values('time').to_julian_date().values.reshape((df.shape[0],1))
    if fitted_scaler is None:
        print('fitting a new time scaler')
        fitted_scaler = MinMaxScaler() 
        scaled_input = fitted_scaler.fit_transform(time_input)
    else:
        print('using a pre-fitted time scaler')
        scaled_input = fitted_scaler.transform(time_input)
    return scaled_input, fitted_scaler

def scale_other_features(df: pd.DataFrame, fitted_scaler: MinMaxScaler = None) -> tuple[np.ndarray, MinMaxScaler]:
    """
    Simple scaling of an input feature dataset (nsamples, nfeatures)
    """
    if fitted_scaler is None:
        print('fitting a new feature scaler')
        fitted_scaler = MinMaxScaler() 
        scaled_input = fitted_scaler.fit_transform(df)
    else:
        print('using a pre-fitted time scaler')
        scaled_input = fitted_scaler.transform(df)
    return scaled_input, fitted_scaler

def singleclass_regression(binary_obs: pd.Series, regressor = LogisticRegression) -> tuple[dict,np.ndarray,MinMaxScaler]:
    """
    Singleclass benchmark, returning the input, scaler, and regressor
    Time (julian-day) is the only input (index of the binary obs), but needs to be min-max scaled. So fitted scaler is returned too
    Can happen on all data (train + validation)
    """
    scaled_input, time_scaler = scale_time(binary_obs)
    # Create and fit the regressor
    lr = regressor() # Initialize
    lr.fit(X = scaled_input, y = binary_obs.values)
    if isinstance(lr, LogisticRegression):
        lr.predict = lambda x: lr.predict_proba(x)[:,-1] # Always the positive probabilistic predictions
    return scaled_input, time_scaler, lr

def multiclass_logistic_regression_coefficients(onehot_obs: pd.DataFrame) -> tuple[dict,np.ndarray,MinMaxScaler]:
    """
    Generates the coefficients for the neural network 
    predicting deviations from the (Logistic) climatological trend in the binary predictand
    Time (julian-day) is the only input (index of the binary obs), but needs to be min-max scaled. So fitted scaler is returned too
    Can happen on all data (train + validation)
    returns a prepared dictionary with coeficient and intercept arrays (2,) with the positive class last
    also returns the scaler
    """
    scaled_input, time_scaler = scale_time(onehot_obs)
    nclasses = onehot_obs.shape[-1]
    coefs = np.repeat(np.nan, nclasses)
    intercepts = np.repeat(np.nan, nclasses)
    for i in range(nclasses): # Loop over the categories
        lr = LogisticRegression() # Create and fit the regressor
        lr.fit(X = scaled_input, y = onehot_obs.iloc[:,i])
        coefs[i] = lr.coef_[0,[0]]
        intercepts[i] = lr.intercept_
    climprobkwargs = dict(coefs = coefs, intercepts = intercepts)
    return climprobkwargs, scaled_input, time_scaler 


if __name__ == '__main__':
    leadtimepool = [4,5,6,7,8] 
    #leadtimepool = 15
    targetname = 'books_paper3-2_tg-ex-q0.75-21D_JJA_45r1_1D_15-t2m-q095-adapted-mean.csv'
    observations, forecast = read_raw_predictand(booksname = targetname, clustid = 9, separation = leadtimepool, dynamic_prediction_too = True)
#    regimename = 'books_paper3-4-regimes_z-anom_JJA_45r1_21D-frequency_ids.csv'
#    regforc, regobs = read_raw_predictor_regimes(booksname = regimename, clustid = slice(None), separation = leadtimepool, observation_too = True)
#    annotate_raw_predictor(regforc, variable = 'z-reg', timeagg = 21, metric = 'freq')
    
    #predictors, forc, obs = prepare_full_set(targetname, ndaythreshold = 3, predictand_cluster = 9, leadtimepool = leadtimepool)
    #obs_test, obs_trainval, g = test_trainval_split(obs, crossval = True)
    #regs = read_raw_predictor_regimes(booksname = 'books_paper3-4-regimes_z-anom_JJA_45r1_21D-frequency_ids.csv', clustid = slice(None), separation = slice(None)) 

    # Test what a trend variable does. Not a trend in probability
    # So perhaps just global mean surface temperature?
#    if True:
#        trend_path = '/nobackup/users/straaten/predsets/tg_monthly_global_mean_surface_only_trend.nc'
#        trend = xr.open_dataarray(trend_path).to_dataframe()  
#        trend.columns = pd.MultiIndex.from_tuples([('tg-anom',31,0,'mean')], names = predictors.columns.names)
#        predictors = predictors.join(trend, how = 'inner')
#    jfilter, jmeasure = filter_predictor_set(predictors, obs, return_measures = True, nmost_important = 3)
#    pfilter, pmeasure = filter_predictor_set(predictors, obs, how = perkins, return_measures = True, nmost_important = 3)
    #obs, forc = read_raw_predictand(name, 9, leadtimepool, True)
    #name = 'books_paper3-3-simple_swvl4-anom_JJA_45r1_7D-roll-mean_1-swvl-simple-mean.csv'
    #predictor = read_raw_predictor_ensmean(name, slice(None), leadtimepool) 
    #annotate_raw_predictor(predictor, 'swvl4',7)

    #file = '/nobackup_1/users/straaten/clusters_cv_spearmanpar_varalpha_strict/precursor.multiagg.parquet'
    """
    Bit of a problem that we don't have the empirical predictors at all leadtimes.
    Actually, the clustids don't correspond over the leadtimes (they can come into existence, or actually switch location).
    So better load only one of the available and if we want to pool leadtimes for empirical then fill with persistent values (the chosen leadtime value for the date that dynamically is predicted with another leadtime in the leadtimepool)
    """
    #df = read_empirical_predictors(file, separation = 5) # 
    
    #g = GroupedGenerator(np.array([1,1,2]))


