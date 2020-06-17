import os
import warnings
warnings.filterwarnings("ignore")
import tensorflow as tf

os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False


class Conv3D_bn(tf.keras.layers.Layer):
    def __init__(self,
            filters     = 32,
            kernel_size = 3,
            strides     = 1,
            padding     = "same",
            data_format = "channels_last",
            activation  = "relu",
            kernel_regularizer = 1e-5,
            use_batchnorm = True,
            **kwargs,
            ):
        super(Conv3D_bn, self).__init__(**kwargs)

        self.filters        = filters
        self.kernel_size    = kernel_size
        self.strides        = strides
        self.padding        = padding
        self.data_format    = data_format
        self.activation     = activation
        self.kernel_regularizer = kernel_regularizer
        self.use_batchnorm  = use_batchnorm

        self.activation_layer = tf.keras.activations.get(self.activation)
        self.conv = tf.keras.layers.Conv3D(
            filters     = self.filters,
            kernel_size = self.kernel_size,
            strides     = self.strides,
            padding     = self.padding,
            data_format = self.data_format,
            kernel_regularizer = tf.keras.regularizers.l2(self.kernel_regularizer),
        )
        if self.use_batchnorm:
            if self.data_format == "channels_first":
                axis = 1
            else:
                axis = -1
            self.batchnorm = tf.keras.layers.BatchNormalization(
                axis=axis, scale=False,)
        else:
            self.batchnorm = None

    def call(self, inputs, training=False):
        x = inputs
        x = self.conv(x)
        if self.batchnorm:
            x = self.batchnorm(x, training=training)
        if self.activation_layer:
            x = self.activation_layer(x)
        return x

    def get_config(self):
        config = {
            "filters":      self.filters,
            "kernel_size":  self.kernel_size,
            "strides":      self.strides,
            "padding":      self.padding,
            "data_format":  self.data_format,
            "activation":   self.activation,
            "kernel_regularizer": self.kernel_regularizer,
            "use_batchnorm": self.use_batchnorm,
        }
        return config

class Pool(tf.keras.layers.MaxPool3D):
    def __init__(self,
            pool_size=2, 
            strides=None,
            padding="same",
            data_format="channels_last",
            **kwargs):
        super(Pool, self).__init__(
            pool_size=(pool_size, pool_size, pool_size),
            strides=strides, padding=padding, data_format=data_format,)

    def call(self, inputs, training=False):
        return super(Pool, self).call(inputs)


class ConvBlock_1(tf.keras.layers.Layer):
    def __init__(self, filters=32, data_format="channels_last", **kwargs):
        super(ConvBlock_1, self).__init__()
        self.filters = filters
        self.data_format = data_format
        self.kwargs = kwargs

        self.conv = Conv3D_bn(
            filters=self.filters, 
            data_format=self.data_format,
            **self.kwargs)
        self.pool = tf.keras.layers.MaxPool3D(
            data_format=self.data_format, padding="same")

    def call(self, inputs, training=False):
        x = inputs
        x = self.conv(x, training=training)
        x = self.pool(x)
        return x

    def get_config(self):
        config = {
            "filters": self.filters,
            "data_format": self.data_format,
        }
        config.update(self.kwargs)
        return config


class ConvBlock_2(tf.keras.layers.Layer):
    def __init__(self, filters=32, **kwargs):
        super(ConvBlock_2, self).__init__()
        self.filters = filters
        self.kwargs = kwargs
        
        self.conv = Conv3D_bn(self.filters, **kwargs)
        self.pool = Conv3D_bn(self.filters, strides=2, **kwargs)

    def call(self, inputs, training=False):
        x = inputs
        x = self.conv(x, training=training)
        x = self.pool(x, training=training)
        return x

    def get_config(self):
        config = {
            "filters": self.filters,
        }
        config.update(self.kwargs)
        return config


class ConvBlock_4(tf.keras.layers.Layer):
    def __init__(self, filters=32, **kwargs):
        super(ConvBlock_4, self).__init__()
        self.filters = filters
        self.kwargs = kwargs

        self.conv_1 = Conv3D_bn(self.filters, **self.kwargs)
        self.conv_2 = Conv3D_bn(self.filters, 1, **self.kwargs)
        self.conv_3 = Conv3D_bn(self.filters, **self.kwargs)
        self.pool = Conv3D_bn(self.filters, strides=2, **kwargs)

    def call(self, inputs, training=False):
        x = inputs
        x = self.conv_1(x, training=training)
        x = self.conv_2(x, training=training)
        x = self.conv_3(x, training=training)
        x = self.pool(x, training=training)
        return x

    def get_config(self):
        config = {
            "filters": self.filters,
        }
        config.update(self.kwargs)
        return config


class Dense(tf.keras.layers.Layer):
    def __init__(self, units, 
            activation="relu",
            kernel_regularizer=1e-5,
            dropout=0., 
            use_batchnorm=False,
            ):
        super(Dense, self).__init__()
        self.units  = units
        self.activation = activation
        self.kernel_regularizer = kernel_regularizer
        self.dropout = dropout
        self.use_batchnorm = use_batchnorm
    
        self.dense = tf.keras.layers.Dense(self.units,
            kernel_regularizer=tf.keras.regularizers.l2(self.kernel_regularizer))
        if self.use_batchnorm:
            self.batchnorm = tf.keras.layers.BatchNormalization(scale=False,)
        else:
            self.batchnorm = None
        self.activation_layer = tf.keras.activations.get(self.activation)
        if self.dropout > 0:
            self.dropout_layer = tf.keras.layers.Dropout(self.dropout)
        else:
            self.dropout_layer = None
        
    def call(self, inputs, training=False):
        x = self.dense(inputs)
        if self.batchnorm:
            x = self.batchnorm(x, training=training)
        if self.activation_layer:
            x = self.activation_layer(x)
        if self.dropout_layer:
            x = self.dropout_layer(x, training=training)
        return x

    def get_config(self):
        config = {
            "units": self.units,
            "activation": self.activation,
            "kernel_regularizer": self.kernel_regularizer,
            "dropout": self.dropout,
            "use_batchnorm": self.use_batchnorm,
        }
        return config


# _custom_objects = {c.__class__.__name__ : c for c in [
#     Conv3D_bn, ConvBlock_1, ConvBlock_2, ConvBlock_4]}

_custom_objects = {
    "Conv3D_bn": Conv3D_bn,
    "ConvBlock_1": ConvBlock_1,
    "ConvBlock_2": ConvBlock_2,
    "ConvBlock_4": ConvBlock_4,
    "Dense": Dense,
}

__layers__ = {
    "pool": Pool,
    "conv": Conv3D_bn,
    "convblock_1": ConvBlock_1,
    "convblock_2": ConvBlock_2,
    "convblock_4": ConvBlock_4,
    "dense": Dense, 
}

def get(name):
    return __layers__[name]