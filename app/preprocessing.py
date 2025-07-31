import pandas as pd
from sklearn.impute import SimpleImputer

def create_features(df: pd.DataFrame, director_avg_map: dict = None, principals_df=None) -> pd.DataFrame:
    # Проверка на пустой DataFrame
    if df.empty:
        return df  # просто возвращаем пустой df

    df = df.copy()

    # --- Основные признаки ---
    df['primaryTitle'] = df['primaryTitle'].fillna('')
    df['description'] = df['primaryTitle']
    df['directors'] = df['directors'].fillna('Unknown')
    df['is_remake'] = df['primaryTitle'].apply(
        lambda x: 1 if isinstance(x, str) and 'remake' in x.lower() else 0
    )

    # --- Жанры ---
    if 'genres' in df.columns:
        genre_cols_before = [col for col in df.columns if col.startswith('genre_')]
        if not genre_cols_before:
            df['genres'] = df['genres'].fillna('Unknown').astype(str)
            genre_dummies = df['genres'].str.get_dummies(sep=',')
            df = df.join(genre_dummies.add_prefix('genre_'))

    # --- Числовые признаки ---
    df['runtimeMinutes'] = pd.to_numeric(df['runtimeMinutes'], errors='coerce')
    df['startYear'] = pd.to_numeric(df['startYear'], errors='coerce')

    # Если после обработки нет строк — возвращаем
    if df.empty:
        return df

    # Заполнение пропусков
    num_imputer = SimpleImputer(strategy='median')
    try:
        df[['runtimeMinutes', 'startYear']] = num_imputer.fit_transform(df[['runtimeMinutes', 'startYear']])
    except ValueError as e:
        if "Found array with 0 sample(s)" in str(e):
            print("⚠️ Внимание: передан пустой массив в SimpleImputer — пропускаем заполнение.")
        else:
            raise e

    # --- Рейтинг режиссёра ---
    if director_avg_map is not None:
        df['director_avg_rating'] = df['directors'].map(director_avg_map)
        global_avg = df['averageRating'].mean() if not df['averageRating'].empty else 6.0
        df['director_avg_rating'] = df['director_avg_rating'].fillna(global_avg)
    else:
        temp_avg = df.groupby('directors')['averageRating'].mean()
        df['director_avg_rating'] = df['directors'].map(temp_avg).fillna(df['averageRating'].mean())

    # --- Актёры ---
    if principals_df is not None and not principals_df.empty:
        actors = principals_df[principals_df['category'] == 'actor'][['tconst', 'nconst']]
        actor_counts = actors.groupby('tconst')['nconst'].count().rename('num_actors')
        df = df.merge(actor_counts, on='tconst', how='left').fillna(0)
    else:
        df['num_actors'] = 0

    return df