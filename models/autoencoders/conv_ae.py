import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np

class ConvAutoencoder(Model):
    def __init__(self, input_shape=(64, 64, 3), latent_dim=128):
        super(ConvAutoencoder, self).__init__()
        self.encoder = self.build_encoder(input_shape, latent_dim)
        self.decoder = self.build_decoder(input_shape, latent_dim)

    def build_encoder(self, input_shape, latent_dim):
        inputs = layers.Input(shape=input_shape)
        x = layers.Conv2D(32, 3, padding='same', activation='relu')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D(2)(x)
        x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D(2)(x)
        x = layers.Conv2D(128, 3, padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D(2)(x)
        x = layers.Conv2D(256, 3, padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        self.shape_before_flattening = tf.keras.backend.int_shape(x)[1:]
        x = layers.Flatten()(x)
        encoded = layers.Dense(latent_dim, activation='relu')(x)
        return Model(inputs, encoded)

    def build_decoder(self, input_shape, latent_dim):
        inputs = layers.Input(shape=(latent_dim,))
        x = layers.Dense(np.prod(self.shape_before_flattening), activation='relu')(inputs)
        x = layers.Reshape(self.shape_before_flattening)(x)
        x = layers.Conv2DTranspose(256, 3, padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2DTranspose(128, 3, padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.UpSampling2D(2)(x)
        x = layers.Conv2DTranspose(64, 3, padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.UpSampling2D(2)(x)
        x = layers.Conv2DTranspose(32, 3, padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.UpSampling2D(2)(x)
        decoded = layers.Conv2D(3, 3, padding='same', activation='sigmoid')(x)
        return Model(inputs, decoded)

    def call(self, inputs):
        encoded = self.encoder(inputs)
        decoded = self.decoder(encoded)
        return decoded

