import pandas as pd
import gzip
from typing import Iterator, Optional

def load_imdb_chunked(
    basics_path: str,
    ratings_path: str,
    crew_path: str,
    chunksize: int = 10_000
) -> Iterator[pd.DataFrame]:
    """
    Загружает IMDb данные по частям, объединяя basics, ratings и crew.
    """
    # Читаем рейтинги целиком (маленький файл)
    df_ratings = pd.read_csv(ratings_path, sep='\t', compression='gzip')
    df_crew = pd.read_csv(crew_path, sep='\t', compression='gzip', dtype={'directors': 'str'})

    # Читаем basics по чанкам
    for chunk in pd.read_csv(basics_path, sep='\t', compression='gzip', chunksize=chunksize):
        merged = chunk.merge(df_ratings, on='tconst', how='inner')
        merged = merged.merge(df_crew[['tconst', 'directors']], on='tconst', how='left')
        yield merged