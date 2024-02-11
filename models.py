import tensorflow as tf

from tensorflow.keras import layers, losses, metrics, Sequential
import tensorflow_recommenders as tfrs

from load_data import BATCH_SIZE
LATENT_DIMS = 128


def get_model(users, movies, movie_data):
    user_model = UserModel(user_ids=users, name="user_model")

    movie_model = MovieModel(movies=movies, name="movie_model")

    retrieval_task = tfrs.tasks.Retrieval(
        metrics=[
            tfrs.metrics.FactorizedTopK(
                movie_data.batch(BATCH_SIZE).map(movie_model)
            )
        ]
    )

    ranking_task = tfrs.tasks.Ranking(
        loss=losses.MeanSquaredError(),
        metrics=[metrics.RootMeanSquaredError()]
           )

    recommender = RecommenderModel(
            user_model,
            movie_model,
            rank_task=ranking_task,
            retrieval_task=retrieval_task,
            )

    return recommender


@tf.keras.saving.register_keras_serializable()
class MovieModel(tf.keras.Model):
    def __init__(self, movies, n_factors=LATENT_DIMS, **kwargs):
        super().__init__(**kwargs)

        self.lookup = Sequential([
            layers.StringLookup(
                vocabulary=movies, mask_token=None, name="movie_id_lookup"),
            layers.Embedding(
                len(movies) + 1, n_factors, name="movie_emb",
                embeddings_regularizer=tf.keras.regularizers.L1L2(0.002, 0.04))
            ])

    def call(self, x, training):
        x = tf.as_string(x)
        x = self.lookup(x)

        return x


@tf.keras.saving.register_keras_serializable()
class UserModel(tf.keras.Model):
    def __init__(self, user_ids, n_factors=LATENT_DIMS, **kwargs):
        super().__init__(**kwargs)
        self.lookup = Sequential([
            layers.StringLookup(
                vocabulary=user_ids, mask_token=None, name="user_id_lookup"),
            layers.Embedding(
                len(user_ids) + 1, n_factors, name="user_emb_lyr",
                embeddings_regularizer=tf.keras.regularizers.L1L2(
                    0.04, 0.0204))
            ], name="user_emb")

    def call(self, x, training):
        x = tf.as_string(x)
        x = self.lookup(x)

        return x


@tf.keras.saving.register_keras_serializable()
class RatingModel(tf.keras.Model):
    def __init__(self, **kwargs):
        super().__init__()
        self.model = Sequential([
            layers.Input((2*LATENT_DIMS,)),
            layers.Dense(256, activation='relu'),
            layers.Dense(64, activation='relu'),
            layers.Dense(1)
        ], **kwargs)

    def call(self, x):
        return self.model(x)


class RecommenderModel(tfrs.Model):
    def __init__(self,
                 user_model: tfrs.Model,
                 movie_model: tfrs.Model,
                 retrieval_task: tfrs.tasks.Retrieval,
                 rank_task: tfrs.tasks.Ranking,
                 retrieval_weight: float = 0.4,
                 rank_weight: float = 0.6,
                 ):
        super().__init__()
        self.user_model = user_model
        self.movie_model = movie_model
        self.rating_model = RatingModel(name="rating_model")
        self.ranking: tfrs.tasks.Task = rank_task
        self.retrieval: tfrs.tasks.Task = retrieval_task

        self.rank_weight = rank_weight
        self.retrieval_weight = retrieval_weight

    def call(self, x):
        user_embedding = self.user_model(x[:, 0])
        movie_embedding = self.movie_model(x[:, 1])

        return user_embedding, movie_embedding, self.rating_model(tf.concat(
            [user_embedding, movie_embedding], axis=1))

    def compute_loss(self, inputs, training=False) -> tf.Tensor:

        x, y = inputs
        user_embedding, movie_embedding, rating_predictions = self(x)

        ranking_loss = self.ranking(
            labels=y, predictions=rating_predictions
            )
        retrieval_loss = self.retrieval(
            user_embedding, movie_embedding
            )

        return (self.rank_weight * ranking_loss +
                self.retrieval_weight * retrieval_loss)
