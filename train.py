import os
import sys
import pathlib
import time
from typing import Union

from tensorflow.keras import optimizers, callbacks
import tensorflow_recommenders as tfrs

from models import get_model, RecommenderModel
from load_data import DataLoader, BATCH_SIZE

EPOCHS = 32
LR = 0.5
MIN_DELTA = 50.0
LR_FACTOR = 0.75

data_loader = DataLoader(pathlib.Path('./data/ml-latest-small/'))


def init_model(name: str, path: Union[str, pathlib.Path]):
    users, movies, movie_data = data_loader.get_model_init_data()
    recommender = get_model(users, movies, movie_data)
    if os.path.exists(f'{path}/{name}.index'):
        print(u"\u001b[32mLoading Weights\033[0m")
        recommender.load_weights(f'{path}/{name}')
    return recommender


def get_retrieval_index(recommender: RecommenderModel):
    brute_force = tfrs.layers.factorized_top_k.BruteForce(
            recommender.user_model)
    brute_force.index_from_dataset(
            data_loader.get_movie_data().batch(BATCH_SIZE)
            .map(lambda id: (id, recommender.movie_model(id)))
        )
    return brute_force


def train(name: str, path: Union[str, pathlib.Path] = 'model_checkpoints'):
    optimizer = optimizers.Adam(LR)
    train_data, val_data = data_loader.load_data(True)
    recommender = init_model(name, path)
    cbs = [
        callbacks.TensorBoard(log_dir=f'./logs/{name}-'+str(time.time())),
        callbacks.TensorBoard(log_dir=f'./logs/{name}'),
        callbacks.ReduceLROnPlateau(
            monitor='total_loss',
            min_delta=MIN_DELTA,
            patience=2,
            factor=LR_FACTOR,
            min_lr=0.00005
            ),
        callbacks.ModelCheckpoint(
            f'{path}/{name}',
            save_weights_only=True
            )
        ]
    recommender.compile(optimizer=optimizer)
    recommender.fit(train_data, epochs=EPOCHS, callbacks=cbs)

    return recommender


if __name__ == "__main__":
    train_run_name = "base" if len(sys.argv) != 2 else sys.argv[-1]

    recommender = train(train_run_name)
