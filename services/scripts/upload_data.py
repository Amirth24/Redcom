import sys
from pathlib import Path
import asyncio
import pandas as pd
from motor.motor_asyncio import AsyncIOMotorClient
from beanie import init_beanie

from ..data_models import Movie, Rating
from datetime import datetime


async def do_everything(data_dir: Path):
    client = AsyncIOMotorClient("mongodb://127.0.0.1:27017/", )

    await init_beanie(
            database=client.redcom,
            document_models=[Rating, Movie])

    movies = pd.read_csv(data_dir/'movies.csv')
    movies = map(
            lambda x: Movie(
                id=str(x[1]['movieId']),
                title=x[1]['title'],
                genres=x[1]['genres'].split('|')
                ),
            movies.iterrows()
        )
    await Movie.insert_many(list(movies))

    ratings = pd.read_csv(data_dir/'ratings.csv')
    ratings = map(
            lambda x: Rating(
                userId=str(x[1]['userId']),
                movieId=str(x[1]['movieId']),
                rating=x[1]['rating'],
                timestamp=datetime.fromtimestamp(x[1]['timestamp'])
                ),
            ratings.iterrows()
        )
    await Rating.insert_many(list(ratings))

if __name__ == "__main__":
    data_dir = sys.argv[-1]
    asyncio.run(do_everything(Path(data_dir)))
