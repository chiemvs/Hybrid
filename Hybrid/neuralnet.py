import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.linear_model import LogisticRegression

"""
Also objective selection should be here
"""

class ClimLogProbLayer(tf.keras.layers.Layer):
    """
    For each class (probability of positive) it requires the fitted coeficients
    if multiple classes, the coefs need to be concatenated into a 1D array of (nclasses,)
    inputs to this layer are numeric values of (normalized?) julian date (t)
    outputs log(p_clim_t) = log(1 / (1 + exp(-(a*t + b))))
    If embedded in another model, the n_out_expected (in terms of nodes) can be supplied for an extra check against nclasses
    """
    def __init__(self, coefs: np.ndarray, intercepts: np.ndarray, activation = 'linear', n_out_expected: int = None, **kwargs):
        super(ClimLogProbLayer, self).__init__(**kwargs)
        self.activation = tf.keras.activations.get(activation)
        self.coefs = coefs # a
        self.intercepts = intercepts # b
        assert (coefs.ndim == 1) and (intercepts.ndim == 1), 'supply coefficient and intercepts in a 1D array each, of equal shape (nclasses,)'
        assert len(coefs) == len(intercepts), 'supply coefficient and intercepts in a 1D array each, of equal shape (nclasses,)'
        self.n_out = len(coefs)
        if not n_out_expected is None:
            assert self.n_out == n_out_expected, f'Climlogprob produces {self.n_out} node from the supplied coefficients, but embedding expects {n_out_expected}'

    def build(self, input_shape):
        # input shape should just be (batchsize,1) because only time is going in.
        def init_a(shape, dtype=None):
            return tf.constant(value= self.coefs, shape=shape, dtype= tf.dtypes.float32)
        def init_b(shape, dtype=None):
            return tf.constant(value= self.intercepts, shape=shape, dtype= tf.dtypes.float32)
        self.a = self.add_weight(name = "a",
                shape=[1,self.n_out], # add an inner dimension to the later matrix multiplication
                initializer = init_a,
                trainable = False)
        self.b = self.add_weight(name = "b",
                shape=[self.n_out,],                                                                                     initializer = init_b,
                trainable = False)

    def call(self, inputs):
        # logarithm of p_clim. So when combined with exponential activation instad of the standard linear, then p_clim
        return self.activation(tf.math.negative(tf.math.log1p(tf.exp(tf.math.negative(tf.add(tf.matmul(inputs, self.a),self.b))))))
    
 
def construct_climdev_model(n_classes: int, n_hidden_layers: int, n_features: int, climprobkwargs = {}):
    """
    Creates a two-branch Classifier model (n_classes)
    Branch one (simple) predicts changing climatological probability from time only
    Branch two (complex) is of depth n_hidden_layers and learns from the other input features.
    N_features is excluding time (supplied separately)
    """
    assert n_classes >= 2, 'Also for the binary case we use two output classes, that are normalized to 1'
    time_input = tf.keras.layers.Input((1,))
    feature_input = tf.keras.layers.Input((n_features,))
    initializer = tf.keras.initializers.Zeros() # initialization of weights should be optimal for the activation function
    x = feature_input
    for i in range(n_hidden_layers):
        x = tf.keras.layers.Dense(units = n_features, activation='elu', kernel_initializer = initializer)(x)
    x = tf.keras.layers.Dense(units = n_classes, activation='elu', kernel_initializer = initializer)(x)
    
    log_p_clim = ClimLogProbLayer(n_out_expected = n_classes,**climprobkwargs)(time_input)
    pre_activation = tf.keras.layers.Add()([log_p_clim, x]) # addition: x + log_p_clim. outputs the logarithm of class probablity. Multiplicative nature seen in e.g. softmax: exp(x + log_p_clim) = exp(x)*p_clim
    prob_dist = tf.keras.layers.Activation('softmax')(pre_activation) # normalized to sum to 1
    
    return tf.keras.models.Model(inputs = [feature_input, time_input], outputs = prob_dist)

def construct_modeldev_model(n_classes: int, n_hidden_layers: int, n_features: int, climprobkwargs = {}):
    """
    Creates a two-branch Classifier model (n_classes)
    Branch one (simple) just passes the (logarithm of) raw model probability per class
    Branch two (complex) is of depth n_hidden_layers and learns from the other input features.
    """
    log_p_raw = tf.keras.layers.Input((n_classes,))
    feature_input = tf.keras.layers.Input((n_features,))
    initializer = tf.keras.initializers.Zeros()
    x = feature_input
    for i in range(n_hidden_layers):
        x = tf.keras.layers.Dense(units = n_features, activation='elu', kernel_initializer = initializer)(x)
    x = tf.keras.layers.Dense(units = n_classes, activation='elu', kernel_initializer = initializer)(x)
    
    output = tf.keras.layers.Add()([log_p_raw, x])
    
    return tf.keras.models.Model(inputs = [feature_input, log_p_raw], outputs = output)

preferred_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = False) # Under the hood logits are still used, but from cached ._keras_logits from e.g. softmax. Logits are no direct outputs of the model (only logarithms would be possible)

earlystop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', min_delta=0, patience=10,
        verbose=1, mode='auto', restore_best_weights=True) 
reducelr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.1, patience=5, verbose=1,
        mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)
