import pathlib

import pandas as pd

import tensorflow as tf
from sklearn.model_selection import train_test_split

BATCH_SIZE = 4096
AUTOTUNE = tf.data.AUTOTUNE
TRAIN_SIZE = 0.8


class DataLoader:
    def __init__(self, path: pathlib.Path):
        self.movies_df = pd.read_csv(path/'movies.csv')
        self.ratings = pd.read_csv(path/'ratings.csv')
        self.users = tf.as_string(self.ratings['userId'].unique())
        self.movies = tf.as_string(self.movies_df['movieId'].unique())
        self.movies_data = tf.data.Dataset.from_tensor_slices(self.movies)

    def get_model_init_data(self):
        return self.users, self.movies, self.movies_data

    def get_movie_data(self):
        return self.movies_data

    def load_data(self, shuffle: bool = False):
        x, y = self.ratings.drop(['rating', 'timestamp'], axis=1), \
            self.ratings['rating'].values.reshape((-1, 1))

        train_x, val_x, train_y, val_y = train_test_split(
            x, y, train_size=TRAIN_SIZE
        )

        train_data = tf.as_string(train_x), \
            tf.constant(train_y, dtype=tf.float32)
        train_data = tf.data.Dataset.from_tensor_slices(train_data)

        val_data = tf.as_string(val_x), tf.constant(val_y, dtype=tf.float32)
        val_data = tf.data.Dataset.from_tensor_slices(val_data)

        train_data = train_data.batch(BATCH_SIZE).cache().prefetch(AUTOTUNE)
        val_data = val_data.batch(BATCH_SIZE).cache().prefetch(AUTOTUNE)

        return train_data, val_data
