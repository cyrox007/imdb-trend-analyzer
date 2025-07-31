from app import Kinovanga

from app.data_loader import load_imdb_chunked
from app.preprocessing import create_features
import pandas as pd

# Создаём экземпляр
kino = Kinovanga()

# Загружаем часть данных для обучения (например, первый чанк)
basics_path = 'data/title.basics.tsv.gz'
ratings_path = 'data/title.ratings.tsv.gz'
crew_path = 'data/title.crew.tsv.gz'

# Читаем один чанк для обучения
chunk_iter = load_imdb_chunked(basics_path, ratings_path, crew_path, chunksize=10000)
df_chunk = next(chunk_iter)  # Берём первый чанк

# Обработка
df_chunk = create_features(df_chunk)

# Обучаем модель
print("Обучаю модель на части данных...")
kino.train(df_chunk)

# Предсказание
rating = kino.predict_rating(
    title="The Batman: Part II",
    director="Мэтт Ривз",
    year=2026,
    runtime=150,
    description="Бэтмен против Смерти. Темный рыцарь в кризисе."
)
print(f"Прогнозируемый рейтинг: {rating}")