import numpy as np
import pandas as pd
import tensorflow as tf

from typing import Callable, Union, List

def model_to_submodel(model: tf.keras.Model, exp_of_log_of_multiplier: bool = False) -> tf.keras.Model:
    """
    Extracts the portion that predicts the logarithm of the multiplier. (before the addition layer)
    output shape equal to model, input shape equal to the model features
    possible the output not the log of multiplier but the multiplier itself
    (which might be nicer interpretation wise)
    """
    output = model.layers[-3].output
    if exp_of_log_of_multiplier:
        output = tf.keras.layers.Activation(tf.math.exp)(output)

    submodel = tf.keras.Model(inputs = [model.layers[0].input], outputs = [output])
    # Make sure not trainable. Also affects the original model
    for l in submodel.layers: 
        l.trainable = False
    submodel.is_submodel = True # To flag that it is pruned and not probabilistic anymore
    return submodel

def assert_model_requirements(model: tf.keras.Model, additional_inputs: np.ndarray):
    """
    Additional inputs (log of forc or scaled time) are required for non-pruned models
    """
    if not hasattr(model, 'is_submodel'):
        model.is_submodel = False
    if not model.is_submodel:
        assert not (additional_inputs is None), "When supplying an un-pruned keras model, also supply the additional time or log-of-forecast inputs"

def gradient(model: tf.keras.Model, feature_inputs: np.ndarray, target_fn: Callable = lambda y: y[:,-1], target_fn_kwargs: dict = {}, additional_inputs: np.ndarray = None, times_input: bool = False) -> np.ndarray:
    """
    Needs correctly scaled inputs (should be an array)
    Gradient is the change in a target with respect to one of the feature inputs, 
    so possible to supply a target function
    to reduce the multiclass prediction to a single number, 
    defaults to using the last positive class prediction 
    but some aggregate or even loss is also possible
    Outputs array with shape of feature_inputs
    Possible to output input times gradient
    """
    assert feature_inputs.ndim == 2, "Only (n_samples, n_features) inputs are allowed"
    x = [tf.convert_to_tensor(feature_inputs)]
    assert_model_requirements(model = model, additional_inputs = additional_inputs)
    if not model.is_submodel:
        x.append(tf.convert_to_tensor(additional_inputs))
    with tf.GradientTape() as tape:
        tape.watch(x[0])
        pred = model(x)
        target = target_fn(pred, **target_fn_kwargs)
    grad = tape.gradient(target, x[0]).numpy()
    if times_input:
        return grad * feature_inputs
    else:
        return grad

def wrap_crossentropy(pred: tf.Tensor, y_true: tf.Tensor):
    """
    Wrap crossentropy such that the prediction can come as first argument (probabilistic prediction)
    """
    return tf.keras.losses.categorical_crossentropy(y_true = y_true, y_pred = pred)

def wrap_mse(pred: tf.Tensor, y_true: tf.Tensor, target_class_index: int):
    """
    From multi-class prediction, compute the mean squared error over a single class (continuous prediction)
    """
    return tf.math.reduce_mean(tf.math.square(pred[:,target_class_index] - y_true))  

def backward_optimization(model: tf.keras.Model, feature_sample: np.ndarray, target_value: Union[np.ndarray,float], target_class_index: int = -1, learning_rate: float = 0.001, iterations: int = 20, additional_inputs: np.ndarray = None):
    """
    The feature sample to start from should be properly normalized
    Function iteratively tunes the feature values for the model prediction to 
    approach a target value. This is either the multiplication value for a single target class 
    (for a pruned model)
    Or for a non-pruned model it is the discrete probability distribution.
    Note that in the case of a non-pruned model it can only tune feature values
    not the additional input
    """
    assert (feature_sample.ndim == 2) and (feature_sample.shape[0] == 1), 'Only possible to optimize a single sample, supply feature_sample with shape (1,n_features)'
    assert_model_requirements(model = model, additional_inputs = additional_inputs)
    if not model.is_submodel:
        # Probabilistic space. We want to minimize CrossEntropy like in the training  
        assert isinstance(target_value, np.ndarray), 'requires a multiclass target because of probabilistic predictions'
        if target_value.ndim == 1:
            target_value = np.expand_dims(target_value, axis = 0)
        target_value = tf.convert_to_tensor(target_value)
        loss_fn = wrap_crossentropy
        loss_fn_kwargs = {'y_true':target_value}
    else:
        # Continuous space (log of multiplier), NOTE: not yet sure if the best choice, perhaps not in log space
        target_value = tf.constant(target_value, dtype = tf.float32)
        loss_fn = wrap_mse
        loss_fn_kwargs = {'y_true':target_value, 'target_class_index':target_class_index}

    # Gradient descend
    history = np.repeat(feature_sample, repeats = iterations + 1, axis = 0)
    for i in range(iterations):
        grad = gradient(model = model, feature_inputs = history[[i],:], target_fn = loss_fn, target_fn_kwargs = loss_fn_kwargs, additional_inputs = additional_inputs)
        history[i+1,:] = history[i,:] - grad * learning_rate
    return history

def combine_input_output(model: tf.keras.Model, feature_inputs: np.ndarray, log_of_raw: np.ndarray, target_class_index: int = -1, feature_names: List[str] = None, index: pd.Index = None) -> pd.DataFrame:
    """
    Create one dataframe where raw model probabilities, correction factors, 
    and post-processed probabilities,
    And input values are all joined together. These should be scaled (because used for prediction)
    (optional) feature names should be a list of strings
    (optional) index should be arraylike or a pd.(Multi)Index
    """
    forc_raw = np.exp(log_of_raw)[:,target_class_index]
    forc_pp = model.predict([feature_inputs,log_of_raw])[:,target_class_index]
    subm = model_to_submodel(model, exp_of_log_of_multiplier = True) 
    correction = subm.predict([feature_inputs])[:,target_class_index]

    total = pd.DataFrame({'f_raw':forc_raw,'multi':correction,'f_cor':forc_pp}, index = index)
    features = pd.DataFrame(feature_inputs, index = index, columns = feature_names) 
    return total.join(features, how = 'left')

def composite_extremes(collection: pd.DataFrame, columnname: str = 'multi',level: float = 0.1) -> pd.DataFrame:
    """
    Subsetting the dataframe using the extreme (quantiles) of a certain column
    Returns mean for above Upper and below lower level and neutral
    """
    level = min(level, 1-level)
    below = collection[columnname].values < collection[columnname].quantile(level)
    above = collection[columnname].values > collection[columnname].quantile(1 - level)
    neutral = np.logical_and(~below,~above)
    return pd.concat([collection.loc[below,:].mean(axis = 0), collection.loc[neutral,:].mean(axis = 0), collection.loc[above,:].mean(axis = 0)], axis =1 , keys = [level, 'neutral',1-level])

