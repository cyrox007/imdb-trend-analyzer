import pandas as pd
from typing import Iterator, Tuple, Optional

def load_imdb_chunked(
    basics_path: str,
    ratings_path: str,
    crew_path: str,
    principals_path: Optional[str] = None,
    chunksize: int = 3000
) -> Iterator[Tuple[pd.DataFrame, Optional[pd.DataFrame]]]:
    """
    Читаем basics по чанкам, а ratings и crew — целиком.
    principals читаем по чанкам, чтобы не убить память.
    """
    # Загружаем ratings и crew целиком (они не такие уж большие)
    print("📥 Загружаю ratings и crew целиком...")
    df_ratings = pd.read_csv(ratings_path, sep='\t', compression='gzip', na_values=r'\N')
    df_crew = pd.read_csv(crew_path, sep='\t', compression='gzip', na_values=r'\N', dtype={'directors': 'str'})
    df_crew = df_crew[['tconst', 'directors']].drop_duplicates()

    # Итератор для principals (если нужен)
    principals_iter = None
    if principals_path:
        principals_iter = pd.read_csv(
            principals_path,
            sep='\t',
            compression='gzip',
            na_values=r'\N',
            chunksize=chunksize
        )

    # Читаем basics по чанкам
    basics_iter = pd.read_csv(
        basics_path,
        sep='\t',
        compression='gzip',
        chunksize=chunksize,
        na_values=r'\N'
    )

    for basics_chunk in basics_iter:
        # Фильтр: только фильмы
        if 'titleType' in basics_chunk.columns:
            basics_chunk = basics_chunk[basics_chunk['titleType'] == 'movie']

        if basics_chunk.empty:
            yield basics_chunk, None
            continue

        # Получаем tconst из текущего чанка
        tconst_chunk = basics_chunk['tconst'].tolist()

        # Фильтруем ratings и crew
        ratings_chunk = df_ratings[df_ratings['tconst'].isin(tconst_chunk)]
        crew_chunk = df_crew[df_crew['tconst'].isin(tconst_chunk)]

        # Объединяем
        merged = basics_chunk.merge(ratings_chunk, on='tconst', how='inner')
        merged = merged.merge(crew_chunk, on='tconst', how='left')
        merged['directors'] = merged['directors'].fillna('Unknown')

        # Получаем чанк principals, если нужен
        principals_chunk = None
        if principals_iter:
            try:
                full_principals_chunk = next(principals_iter)
                principals_chunk = full_principals_chunk[full_principals_chunk['tconst'].isin(tconst_chunk)]
            except StopIteration:
                principals_chunk = None

        yield merged, principals_chunk