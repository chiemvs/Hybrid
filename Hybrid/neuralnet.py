import numpy as np
import tensorflow as tf

from typing import Callable

class BrierScore(tf.keras.metrics.Metric):
    """
    Accumulates the quadratic distance (p_i - o_i)**2 across batches
    IMPORTANT: assumes that the model predictions are multi-class arrays (None, classes) 
    of which the last one [:,-1] is the positive class (just like the one hot encoded y_true)
    But another class-index can also be supplied 
    """
    def __init__(self, name = 'brier', class_index = -1, **kwargs):
        super(BrierScore, self).__init__(name = name, **kwargs)
        self.total = self.add_weight(name="quadratic_accumulate", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")
        self.class_index = class_index

    def update_state(self, y_true, y_pred, sample_weight = None):
        y_true = tf.cast(y_true[:,self.class_index], 'float32')
        y_pred = tf.cast(y_pred[:,self.class_index], 'float32')
        quadr_dist = tf.math.squared_difference(y_pred, y_true) # Take the last predicted class
        self.total.assign_add(tf.reduce_sum(quadr_dist))
        self.count.assign_add(tf.cast(tf.size(y_true),'float32'))
    
    def result(self):
        return tf.divide(self.total, self.count)

    def reset_state(self):
        self.count.assign(0)
        self.total.assign(0)

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
    
 
def construct_climdev_model(n_classes: int, n_hidden_layers: int, n_features: int, climprobkwargs = {}, n_hiddenlayer_nodes: int = 5):
    """
    Creates a two-branch Classifier model (n_classes)
    Branch one (simple) predicts changing climatological probability from time only
    Branch two (complex) is of depth n_hidden_layers and learns from the other input features.
    N_features is excluding time (supplied separately)
    """
    assert n_classes >= 2, 'Also for the binary case we use two output classes, that are normalized to 1'
    time_input = tf.keras.layers.Input((1,))
    feature_input = tf.keras.layers.Input((n_features,))
    initializer = tf.keras.initializers.RandomNormal() # initialization of weights should be optimal for the activation function
    #initializer = tf.keras.initializers.Zeros() # initialization of weights should be optimal for the activation function
    x = feature_input
    for i in range(n_hidden_layers):
        x = tf.keras.layers.Dense(units = n_hiddenlayer_nodes, activation='elu', kernel_initializer = initializer)(x) # was units = n_features, but not logical to scale with predictors. 10 is choice in Scheuerer 2020 (with 20 outputs)
    x = tf.keras.layers.Dense(units = n_classes, activation='elu', kernel_initializer = initializer)(x)
    
    log_p_clim = ClimLogProbLayer(n_out_expected = n_classes,**climprobkwargs)(time_input)
    pre_activation = tf.keras.layers.Add()([log_p_clim, x]) # addition: x + log_p_clim. outputs the logarithm of class probablity. Multiplicative nature seen in e.g. softmax: exp(x + log_p_clim) = exp(x)*p_clim
    prob_dist = tf.keras.layers.Activation('softmax')(pre_activation) # normalized to sum to 1
    
    return tf.keras.models.Model(inputs = [feature_input, time_input], outputs = prob_dist)

def construct_modeldev_model(n_classes: int, n_hidden_layers: int, n_features: int, n_hiddenlayer_nodes: int = 5 ):
    """
    Creates a two-branch Classifier model (n_classes)
    Branch one (simple) just passes the (logarithm of) raw model probability per class
    Branch two (complex) is of depth n_hidden_layers and learns from the other input features.
    """
    assert n_classes >= 2, 'Also for the binary case we use two output classes, that are normalized to 1'
    log_p_raw = tf.keras.layers.Input((n_classes,)) # negative class should be the first node (in case of binary)
    feature_input = tf.keras.layers.Input((n_features,))
    initializer = tf.keras.initializers.RandomNormal() # initialization of weights should be optimal for the activation function
    #initializer = tf.keras.initializers.Zeros()
    x = feature_input
    for i in range(n_hidden_layers):
        x = tf.keras.layers.Dense(units = n_hiddenlayer_nodes, activation='elu', kernel_initializer = initializer)(x) # was units = n_features, but not logical to scale with predictors. 10 is choice in Scheuerer 2020 (with 20 outputs)
    x = tf.keras.layers.Dense(units = n_classes, activation='elu', kernel_initializer = initializer)(x)
    
    pre_activation = tf.keras.layers.Add()([log_p_raw, x])
    prob_dist = tf.keras.layers.Activation('softmax')(pre_activation) # normalized to sum to 1
    
    return tf.keras.models.Model(inputs = [feature_input, log_p_raw], outputs = prob_dist)

preferred_loss = tf.keras.losses.CategoricalCrossentropy(from_logits = False) # Under the hood logits are still used, but from cached ._keras_logits from e.g. softmax. Logits are no direct outputs of the model (only logarithms would be possible)

def earlystop(patience: int = 10, monitor: str = 'val_loss'):
    return tf.keras.callbacks.EarlyStopping(
        monitor=monitor, min_delta=0, patience=patience,
        verbose=1, mode='auto', restore_best_weights=True) 

reducelr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.1, patience=5, verbose=1,
        mode='auto', min_delta=0.0001, cooldown=0, min_lr=1e-6)

class ConstructorAndCompiler(object):
    def __init__(self, construct_func: Callable = construct_modeldev_model, construct_kwargs: dict = dict(n_classes = 2, n_hidden_layers = 0, n_features = 10), compile_kwargs: dict = dict(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))):
        """
        construct func is one of the degined above (construct_modeldev_model, construct_climdev_model)
        Default arguments as an example
        """
        self.construct_func = construct_func
        self.construct_kwargs = construct_kwargs
        self.compile_kwargs = compile_kwargs

    def fresh_model(self):
        model = self.construct_func(**self.construct_kwargs)
        model.compile(loss=preferred_loss, **self.compile_kwargs)
        return model

