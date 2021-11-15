import numpy as np
import tensorflow as tf

from typing import Callable, Union, List

def strip_modeldev(model: tf.keras.Model, exp_of_log_of_multiplier: bool = False) -> tf.keras.Model:
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
    return submodel

def gradient(submodel: tf.keras.Model, feature_inputs: np.ndarray, target_fn: Callable = lambda i: i[:,-1], target_fn_kwargs: dict = {}, times_input: bool = False) -> np.ndarray:
    """
    Needs correctly scaled inputs (should be an array)
    Gradient is the change in a target with respect to some input, 
    so possible to supply a target function
    to reduce the multiclass prediction to a single number, 
    defaults to using the last positive class but some aggregate or even loss is also possible
    Outputs array with shape of feature_inputs
    Possible to output input times gradient
    """
    assert feature_inputs.ndim == 2, "Only (n_samples, n_features) inputs are allowed"
    x = tf.convert_to_tensor(feature_inputs)
    with tf.GradientTape() as tape:
        tape.watch(x)
        pred = submodel(x)
        target = target_fn(pred, **target_fn_kwargs)
    grad = tape.gradient(target, x).numpy()
    if times_input:
        return grad * feature_inputs
    else:
        return grad

def backwards_optimization(submodel: tf.keras.Model, learning_rate: float = 0.001, iterations: int = 200):
    pass

def combine_input_output(model: tf.keras.Model, feature_inputs: np.ndarray, log_of_raw: np.ndarray, target_class_index: int = -1):
    """
    Create one dataframe where raw model probabilities, correction factors, 
    and post-processed probabilities,
    And input values are all joined together. These can be scaled or unscaled
    """
    forc_raw = np.exp(log_of_raw)[:,target_class_index]
    forc_pp = model.predict([feature_inputs,log_of_raw])[:,target_class_index]
    subm = strip_modeldev(model, exp_of_log_of_multiplier = True) 
    correction = subm.predict([feature_inputs])[:,target_class_index]

