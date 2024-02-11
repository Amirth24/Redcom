from datetime import datetime
from typing import List
from beanie import Document, Indexed


class Rating(Document):
    userId: str
    movieId: str
    rating: Indexed(float)
    timestamp: datetime


class Movie(Document):
    id: str
    title: str
    genres: List[str]
