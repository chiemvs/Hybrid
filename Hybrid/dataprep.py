import sys
import os
import warnings
import pandas as pd
import numpy as np
try:
    import tensorflow as tf
except ImportError:
    pass

from copy import deepcopy
from typing import Union, Callable, Tuple, List
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

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
        warnings.warn(f'using alpha {alpha} for forecast categorization')
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
    def __next__(self) -> Tuple[np.ndarray, np.ndarray]:
        if self.ncalls < self.ngroups:
            groupid = self.groupids[self.ncalls]
            wheretrue = self.groups == groupid 
            self.ncalls += 1
            return ~wheretrue, wheretrue # train, val
        raise StopIteration()
    def reset(self):
        self.ncalls = 0
    def __copy__(self):
        return type(self)(self.groups)
    def __deepcopy__(self, memo): # based on https://stackoverflow.com/questions/4794244/how-can-i-create-a-copy-of-an-object-in-python
        id_self = id(self)
        _copy = memo.get(id_self)
        if _copy is None:
            _copy = type(self)(deepcopy(self.groups, memo))
            memo[id_self] = _copy
        return _copy

class SingleGenerator(object):
    """ Yields only once the true indices that it has been supplied with and its inverse """
    def __init__(self, whereval: np.ndarray):
        assert (whereval.ndim == 1) and (whereval.dtype == bool), 'only a 1D array with boolean values can be used to yield those as indices'
        self.whereval = whereval
        self.ncalls = 0
    def __iter__(self):
        return self
    def __next__(self) -> Tuple[np.ndarray, np.ndarray]:
        if self.ncalls < 1:
            self.ncalls += 1
            return ~self.whereval, self.whereval
        raise StopIteration()
    def reset(self):
        self.ncalls = 0
    def __copy__(self):
        return type(self)(self.whereval)
    def __deepcopy__(self, memo): # based on https://stackoverflow.com/questions/4794244/how-can-i-create-a-copy-of-an-object-in-python
        id_self = id(self)
        _copy = memo.get(id_self)
        if _copy is None:
            _copy = type(self)(deepcopy(self.whereval, memo))
            memo[id_self] = _copy
        return _copy

def test_trainval_split(df: Union[pd.Series, pd.DataFrame], crossval: bool = False, nfolds: int = 3, balanced: bool = True) -> Tuple[Union[pd.Series, pd.DataFrame],Union[GroupedGenerator,SingleGenerator]]:
    """
    Hardcoded train/val test split, supplied dataset should be indexed by time
    Returns a single test set and a single combined train/validation set,
    plus a generator that can be called for the right indices for the latter 
    crossval = True with nfolds = 4 will lead to 4 times different indices, which are split by year  
    Option to do a predetermined (hardcoded) crossval that balances hot and cold years over the folds.
    Only works with three folds
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
    elif balanced:
        assert nfolds == 3, 'balanced crossvalidation only done for three folds'
        #division = pd.Series([0,1,2,0,1,2,0,2,1,0,1,2,0,1,2,99,99,99,2,1,0,99], index = pd.RangeIndex(1998,2020, name = 'year')) #Based on excel balancing
        division = pd.Series([ 0,1,2,1,1,2,2,0,1,0,0,1,1,2,0,99,2,99,99,2,0,99], index = pd.RangeIndex(1998,2020, name = 'year')) # Generated with generate_balanced_kfold with forecasts > 7 hot days in 21 day period, leadtime = 19-21. 
        division = division.reindex(df.index.get_level_values('time').year).values
        testset = df.loc[division == 99] # Actually a slightly different test set, and array indexing as opposed to slicing
        trainvalset = df.loc[division != 99]
        generator = GroupedGenerator(groups = division[division != 99])
    else:
        years = trainvalset.index.get_level_values('time').year
        unique_years = years.unique()
        assert nfolds <= len(unique_years), 'More folds than unique years requested. Unable to split_on_year.'
        groupsize = int(np.floor(len(unique_years) / nfolds)) # Maximum even groupsize, except for the last fold, if unevenly divisible then last fold gets groupsize + the remainder.
        groups = (years - years.min()).map(lambda year: min(year // groupsize, nfolds-1))  
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
                #print('too little in',k)
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

def scale_time(df: Union[pd.Series,pd.DataFrame], fitted_scaler: MinMaxScaler = None) -> Tuple[np.ndarray,MinMaxScaler]:
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

def scale_other_features(df: Union[pd.DataFrame,np.ndarray], fitted_scaler: MinMaxScaler = None) -> Tuple[np.ndarray, MinMaxScaler]:
    """
    Simple scaling of an input feature dataset (nsamples, nfeatures)
    """
    if fitted_scaler is None:
        print('fitting a new feature scaler')
        fitted_scaler = MinMaxScaler() 
        scaled_input = fitted_scaler.fit_transform(df)
    else:
        print('using a pre-fitted feature scaler')
        scaled_input = fitted_scaler.transform(df)
    return scaled_input, fitted_scaler

def singleclass_regression(binary_obs: pd.Series, regressor = LogisticRegression) -> Tuple[dict,np.ndarray,MinMaxScaler]:
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

def multiclass_logistic_regression_coefficients(onehot_obs: pd.DataFrame) -> Tuple[dict,np.ndarray,MinMaxScaler]:
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

def generate_balanced_kfold(f_probs: pd.Series, shuffle = False) -> np.ndarray:
    """
    Requirements:
    - balanced random sample of high, normal, and low forecast probabilities
    - probability classes determined by terciles
    - test data has to be picked in 2013 and beyond (separate in 2)
    Produces:
    - division into 4 groups. 3 train-val folds, 1 test group
    """
    if not f_probs.index.dtype == np.int64: 
        f_probs = f_probs.groupby(f_probs.index.get_level_values('time').year).mean()

    hot = f_probs > f_probs.quantile(0.666)
    cold = f_probs < f_probs.quantile(0.333)
    normal = np.logical_and(~hot,~cold)
    classes = pd.concat([cold*10, normal*20, hot*30], axis = 1).max(axis = 1) 
    groups = classes.copy() # Will be overwritten 
    print('generating test group')
    post_2013 = classes.loc[2013:]
    skf = StratifiedKFold(n_splits = 2, shuffle = shuffle)
    _, testgroup = next(skf.split(post_2013, post_2013))
    groups.loc[2013:,].iloc[testgroup] = 99 # Test part
    print('generating trainval groups')
    available = groups.values != 99 
    trainvalgroups = groups.iloc[available].copy()
    skf = StratifiedKFold(n_splits = 3, shuffle = shuffle)
    for i, (_, valgroup) in enumerate(skf.split(classes.iloc[available], classes.iloc[available])):
        trainvalgroups.iloc[valgroup] = i
    groups.iloc[available] = trainvalgroups
    return classes, groups


if __name__ == '__main__':
    yearset = np.arange(1998,2020)
    # cold = -1, neutral = 0, hot = 1, cold = 0 Based on the mean forecast probability in that season
    classes = np.array([-1,-1,0,0,0,1,0,1,1,-1,-1,0,0,0,-1,0,-1,1,0,0,1,1]) 
    division = pd.Series(classes, index = yearset)
    # Classes with based on forcast terciles:
    classes2 = np.array([0,0,-1,-1,-1,1,-1,1,1,-1,-1,0,1,0,0,-1,0,1,0,1,1,1]) 
    division2 = pd.Series(classes2, index = yearset)

    # First 2 folds in 2013 + to split trainval from test there
    # Then in all remaining years. StratifiedKFold
