import numpy as np
import pandas as pd

from typing import Union, Callable, List, Tuple

from .dataprep import  GroupedGenerator, SingleGenerator, scale_other_features

"""
The place with functionality for 
- objective predictor selection
- hyperparameter optimization
which will be arranged in scripts
"""
def ranked_prob_score(forecasts : np.ndarray, observations: np.ndarray, weights: np.ndarray = None):
    """
    Both arrays should be (n_samples, n_classes). For deterministic observations
    the array is one-hot encoded. For fuzzy (probabilistic observations)
    It should be a probability distribution summing up to one (over the n_classes axis)
    weights is optional and (n_samples,)
    """
    distances = (forecasts - observations)**2
    distances_over_classes = distances.sum(axis = 1) # Over classes
    rps = np.average(distances_over_classes, axis = 0, weights = weights) # Over samples
    return rps

def multi_fit_single_eval(constructor, X_trainval: Tuple[np.ndarray,np.ndarray], y_trainval: np.ndarray, generator: Union[GroupedGenerator,SingleGenerator], fit_kwargs: dict = dict(batch_size = 32, epochs = 200), score_func: Callable = ranked_prob_score, return_predictions: bool = False, scale_cv_mode: bool = False) -> float:
    """
    Initialized constructer will supply on demand new freshly inilizated models
    this function fits as many (neural) models as the generator generates train/validation subsets 
    X_trainval is a list of the two inputs required to the neural nets 
    either [predictors, time] for climdev or [predictors, raw_log_probs] for modeldev
    Concatenates the (potential multiclass) predictions of the models and evaluates with a scoring func
    """
    predictions = np.full(y_trainval.shape, np.nan)
    histories = []
    for trainind, valind in generator: # Entering the crossvalidation
        X_train, X_extra_train = X_trainval[0][trainind,:], X_trainval[-1][trainind,...]
        X_val, X_extra_val = X_trainval[0][valind,:], X_trainval[-1][valind,...]
        if scale_cv_mode:
            X_train, feature_scaler = scale_other_features(X_train)
            X_val, feature_scaler = scale_other_features(X_val, fitted_scaler = feature_scaler)
        y_train = y_trainval[trainind,...]
        y_val = y_trainval[valind,...]
        model = constructor.fresh_model() # With neural nets it is important that we re-initialize
        history = model.fit(x = [X_train, X_extra_train], 
                y = y_train, 
                validation_data=([X_val, X_extra_val], y_val), 
                **fit_kwargs)
        predictions[valind,...] = model.predict([X_val, X_extra_val])
        histories.append(history)
    if isinstance(generator, SingleGenerator): # Subsetting becaus we do no crossval to make predictions at all datapoints
        score = score_func(predictions[valind,...], y_trainval[valind,...]) 
    else:
        score = score_func(predictions, y_trainval) 
    if return_predictions:
        return score, histories, predictions
    else:
        return score, histories

def multi_fit_multi_eval(constructor, X_trainval: Tuple[np.ndarray,np.ndarray], y_trainval: np.ndarray, generator: Union[GroupedGenerator,SingleGenerator], fit_kwargs: dict = dict(batch_size = 32, epochs = 200), scale_cv_mode: bool = False) -> pd.DataFrame:
    """
    Initialized constructer will supply on demand new freshly inilizated models
    this function fits as many (neural) models as the generator generates train/validation subsets 
    X_trainval is a list of the two inputs required to the neural nets 
    either [predictors, time] for climdev or [predictors, raw_log_probs] for modeldev
    """
    nscores = 1 # namely the loss function
    if 'metrics' in constructor.compile_kwargs:
        nscores += len(constructor.compile_kwargs['metrics'])
    if isinstance(generator, SingleGenerator):
        results = pd.DataFrame(np.nan, index = pd.MultiIndex.from_product([[0], ['train','val']], names = ['fold','part']), columns = pd.RangeIndex(nscores))
    else:
        results = pd.DataFrame(np.nan, index = pd.MultiIndex.from_product([generator.groupids, ['train','val']], names = ['fold','part']), columns = pd.RangeIndex(nscores))
    for i, (trainind, valind) in enumerate(generator): # Entering the crossvalidation
        X_train, X_extra_train = X_trainval[0][trainind,:], X_trainval[-1][trainind,...]
        X_val, X_extra_val = X_trainval[0][valind,:], X_trainval[-1][valind,...]
        if scale_cv_mode:
            X_train, feature_scaler = scale_other_features(X_train)
            X_val, feature_scaler = scale_other_features(X_val, fitted_scaler = feature_scaler)
        y_train = y_trainval[trainind,...]
        y_val = y_trainval[valind,...]
        model = constructor.fresh_model() # With neural nets it is important that we re-initialize
        model.fit(x = [X_train, X_extra_train], 
                y = y_train, 
                validation_data=([X_val, X_extra_val], y_val), 
                **fit_kwargs)
        results.loc[(i,'train'),:] = model.evaluate([X_train, X_extra_train], y_train)
        results.loc[(i,'val'),:] = model.evaluate([X_val, X_extra_val], y_val)
    return results
