import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, InputLayer, Flatten, Conv2DTranspose, Reshape, LSTM
from tensorflow.keras import Sequential

class AE(tf.keras.Model):
    def __init__(self, latent_dim, size):
        super(AE, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = Sequential(
        [
            InputLayer(input_shape=size + (1, )),
            Conv2D(
                filters=32, kernel_size=3, strides=2, activation='relu'),
            Conv2D(
                filters=64, kernel_size=3, strides=2, activation='relu'),
            Flatten(),
            # No activation
            Dense(latent_dim)
        ]
    )

        self.decoder = tf.keras.Sequential(
        [
            InputLayer(input_shape=(latent_dim,)),
            Dense(units=16*16*32, activation='relu'),
            Reshape(target_shape=(16, 16, 32)),
            Conv2DTranspose(
                filters=64, kernel_size=3, strides=2, padding='same',
                activation='relu'),
            Conv2DTranspose(
                filters=32, kernel_size=3, strides=2, padding='same',
                activation='relu'),
            # No activation
            Conv2DTranspose(
                filters=1, kernel_size=3, strides=1, padding='same',
                activation='sigmoid'),
        ]
    )
    
    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

class LSTM_predictor(tf.keras.Model):
    
    def __init__(self, hidden_size):
        super(LSTM_predictor, self).__init__()

        self.model = Sequential(
            [   
                LSTM(hidden_size),
                Dense(hidden_size // 2,  activation='relu'),
                Dense(1, activation='sigmoid')
            ]
        )

    def predict(self, x):
        return self.model(x)
