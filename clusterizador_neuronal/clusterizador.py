import numpy as np
from sklearn.datasets import load_digits
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras


def build_autoencoder(input_dim, encoding_dim):
    input_layer = keras.layers.Input(shape=(input_dim,))
    encoded = keras.layers.Dense(encoding_dim, activation='relu')(input_layer)
    decoded = keras.layers.Dense(input_dim, activation='sigmoid')(encoded)
    autoencoder = keras.models.Model(inputs=input_layer, outputs=decoded)
    encoder = keras.models.Model(inputs=input_layer, outputs=encoded)
    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder, encoder


def cluster_data(x, n_clusters=10, encoding_dim=32, epochs=20, batch_size=32):
    scaler = MinMaxScaler()
    x_scaled = scaler.fit_transform(x)
    autoencoder, encoder = build_autoencoder(x_scaled.shape[1], encoding_dim)
    autoencoder.fit(x_scaled, x_scaled, epochs=epochs, batch_size=batch_size, verbose=0)
    encoded_x = encoder.predict(x_scaled)
    kmeans = KMeans(n_clusters=n_clusters, n_init='auto')
    labels = kmeans.fit_predict(encoded_x)
    return labels


def main():
    digits = load_digits()
    x = digits.data
    labels = cluster_data(x)
    print('Cluster labels for the first 10 samples:', labels[:10])


if __name__ == "__main__":
    main()
