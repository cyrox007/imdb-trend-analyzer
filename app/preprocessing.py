import pandas as pd
from sklearn.impute import SimpleImputer

def create_features(df: pd.DataFrame, director_avg_map: dict = None) -> pd.DataFrame:
    """
    Создаёт признаки для модели.
    :param df: исходный DataFrame
    :param director_avg_map: словарь {режиссёр: средний рейтинг}
    """
    df = df.copy()

    # Обработка пропусков
    df['primaryTitle'] = df['primaryTitle'].fillna('')
    df['description'] = df['primaryTitle']  # Пока используем название как описание
    df['directors'] = df['directors'].fillna('Unknown')

    # Признак "ремейк"
    df['is_remake'] = df['primaryTitle'].apply(
        lambda x: 1 if isinstance(x, str) and 'remake' in x.lower() else 0
    )

    # Числовые признаки
    df['runtimeMinutes'] = pd.to_numeric(df['runtimeMinutes'], errors='coerce')
    df['startYear'] = pd.to_numeric(df['startYear'], errors='coerce')

    # Заполнение пропусков
    num_imputer = SimpleImputer(strategy='median')
    df[['runtimeMinutes', 'startYear']] = num_imputer.fit_transform(df[['runtimeMinutes', 'startYear']])

    # Добавляем средний рейтинг режиссёра
    if director_avg_map is not None:
        df['director_avg_rating'] = df['directors'].map(director_avg_map)
        # Заполняем пропуски — средним по датасету
        global_avg = df['averageRating'].mean()
        df['director_avg_rating'] = df['director_avg_rating'].fillna(global_avg)
    else:
        # Если нет карты — создаём из текущего чанка (для первого чанка)
        temp_avg = df.groupby('directors')['averageRating'].mean()
        df['director_avg_rating'] = df['directors'].map(temp_avg).fillna(df['averageRating'].mean())

    return df