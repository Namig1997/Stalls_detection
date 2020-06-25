import warnings
warnings.filterwarnings("ignore")
import tensorflow as tf
from .layers import (
    Conv3D_bn, Pool,
    ConvBlock_1, ConvBlock_2,
    ConvBlock_4, Dense,
)

def get_simple_model(input_shape=(32, 32, 32)):
    args = {}
    model = tf.keras.models.Sequential([
        tf.keras.layers.Reshape(input_shape + (1,)),
        ConvBlock_1(32, **args),
        ConvBlock_1(64, **args),
        ConvBlock_1(64, **args),
        ConvBlock_1(64, **args),
        Conv3D_bn(64, kernel_size=2, padding="valid", **args),
        tf.keras.layers.Reshape((64,)),
        Dense(128,),
        Dense(1, activation=tf.keras.activations.sigmoid),
    ])
    return model


def get_simple_model_2(input_shape=(32, 32, 32)):
    args = {}
    model = tf.keras.models.Sequential([
        tf.keras.layers.Reshape(input_shape + (1,)),
        ConvBlock_2(32, **args),
        ConvBlock_2(64, **args),
        ConvBlock_4(64, **args),
        ConvBlock_4(64, **args),
        Conv3D_bn(128, kernel_size=2, padding="valid", **args),
        tf.keras.layers.Reshape((128,)),
        Dense(512, dropout=0.5),
        Dense(1, activation=tf.keras.activations.sigmoid),
    ])
    return model