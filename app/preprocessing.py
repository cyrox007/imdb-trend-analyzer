# kinovanga/preprocessing.py
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Создаёт признаки: ремейк, очистка, средний рейтинг режиссёра"""
    df = df.copy()

    # Обработка пропусков
    df['primaryTitle'] = df['primaryTitle'].fillna('')
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

    # Средний рейтинг режиссёра
    director_avg = df.groupby('directors')['averageRating'].mean()
    df['director_avg_rating'] = df['directors'].map(director_avg).fillna(df['averageRating'].mean())

    return df