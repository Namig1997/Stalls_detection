import warnings
warnings.filterwarnings("ignore")
import tensorflow as tf
from .layers import (
    Conv3D_bn, Pool,
    ConvBlock_1, ConvBlock_2,
    ConvBlock_4, Dense,
)

def get_simple_model(input_shape=(32, 32, 32), version=0, **kwargs):
    input_shape = tuple(input_shape)
    args = {}
    args.update(kwargs)
    if input_shape == (64, 64, 64):
        model = tf.keras.models.Sequential([
            tf.keras.layers.Reshape(input_shape + (1,)),
                # input_shape=input_shape,),
            ConvBlock_2(32, **args),
            ConvBlock_2(32, **args),
            ConvBlock_2(64, **args),
            ConvBlock_2(64, **args),
            ConvBlock_4(64, **args),
            Conv3D_bn(128, kernel_size=2, padding="valid", **args),
            tf.keras.layers.Reshape((128,)),
            Dense(512, dropout=0.5),
            Dense(1, activation=tf.keras.activations.sigmoid),
        ])
    else:
        if input_shape == (32, 32, 32):
            kernel_size_last = 2
        elif input_shape == (48, 48, 48):
            kernel_size_last = 3
        if version == 0:
            model = tf.keras.models.Sequential([
                tf.keras.layers.Reshape(input_shape + (1,)),
                    # input_shape=input_shape,),
                ConvBlock_1(32, **args),
                ConvBlock_1(64, **args),
                ConvBlock_1(64, **args),
                ConvBlock_1(64, **args),
                Conv3D_bn(64, kernel_size=kernel_size_last, 
                    padding="valid", **args),
                tf.keras.layers.Reshape((64,)),
                Dense(128,),
                Dense(1, activation=tf.keras.activations.sigmoid),
            ])
        elif version == 1:
            model = tf.keras.models.Sequential([
                tf.keras.layers.Reshape(input_shape + (1,)),
                    # input_shape=input_shape,),
                ConvBlock_2(32, **args),
                ConvBlock_2(64, **args),
                ConvBlock_4(64, **args),
                ConvBlock_4(64, **args),
                Conv3D_bn(128, kernel_size=kernel_size_last, 
                    padding="valid", **args),
                tf.keras.layers.Reshape((128,)),
                Dense(512, dropout=0.5),
                Dense(1, activation=tf.keras.activations.sigmoid),
            ])
        elif version == 2:
            model = tf.keras.models.Sequential([
                tf.keras.layers.Reshape(input_shape + (1,)),
                ConvBlock_2(32, **args),
                ConvBlock_2(64, **args),
                ConvBlock_4(64, **args),
                ConvBlock_4(64, **args),
                Conv3D_bn(128, kernel_size=kernel_size_last, 
                    padding="valid", **args),
                tf.keras.layers.Reshape((128,)),
                Dense(128),
                Dense(1, activation=tf.keras.activations.sigmoid),
            ])
    return model