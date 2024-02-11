import argparse
import tensorflow as tf
from models import RatingModel


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('model_root', help='Model Root')
    parser.add_argument(
            'userId',
            help='User Id to recommend movies to user')

    args = parser.parse_args()
    # Retrieve movies for user
    # retrieve_model = tfrs.layers.factorized_top_k.BruteForce()
    retrieve_model = tf.keras.models.load_model(args.model_root + '/retrieval')
    # retrieve_model.load_weights(args.model_root + '/retrieval_index')

    print(retrieve_model(['42']))
    rating_model = RatingModel()

    rating_model.load_weights(args.model_root + '/rating')
