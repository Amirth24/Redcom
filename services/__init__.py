import pathlib
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from train import init_model, get_retrieval_index
from .data_models import Movie, Rating
from motor.motor_asyncio import AsyncIOMotorClient
from beanie import init_beanie

app = FastAPI()


async def init_db():
    client = AsyncIOMotorClient(
        'mongodb://127.0.0.1'
        )
    await init_beanie(
        database=client.redcom,
        document_models=[
            Movie, Rating
            ]
        )
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def start_db():
    await init_db()

model_root = "model_checkpoints"
recommender = init_model('base', pathlib.Path(model_root))
retrieve_index = get_retrieval_index(recommender)


@app.get('/get_redcom/{user_id}')
async def get_redcom(user_id: str):
    _, retrieved_movies = retrieve_index([user_id])

    user_retrieved_movies = np.expand_dims(
            retrieved_movies.numpy(), axis=-1).squeeze()
    user_tiles = np.tile([user_id], 10)
    _, _, rating = recommender(
            np.dstack([user_tiles, user_retrieved_movies]).squeeze())

    top_movies = zip(
            rating.numpy().flatten().astype(float),
            retrieved_movies.numpy().flatten().astype(str))

    top_movies = sorted(list(top_movies), reverse=True)

    result = []
    print(top_movies)
    for rating, movie in top_movies:
        mov = await Movie.get(movie)
        result.append([rating, mov])

    return {"top_movies": result}
