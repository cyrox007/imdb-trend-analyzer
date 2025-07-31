from app import Kinovanga

from app.data_loader import load_imdb_chunked
from app.preprocessing import create_features
import pandas as pd

# Пути к данным
basics_path = 'data/title.basics.tsv.gz'
ratings_path = 'data/title.ratings.tsv.gz'
crew_path = 'data/title.crew.tsv.gz'

# Создаём и обучаем модель
kino = Kinovanga()
kino.train_on_chunks(
    basics_path=basics_path,
    ratings_path=ratings_path,
    crew_path=crew_path,
    chunksize=10000,
    max_chunks=20  # Увеличь, если хватает времени
)

# Сохраняем (опционально)
kino.predictor.save("models/kinovanga_sgd.joblib")

# Предсказание
rating = kino.predict_rating(
    title="The Batman: Part II",
    director="Мэтт Ривз",
    year=2026,
    runtime=150,
    description="Бэтмен против Смерти. Темный рыцарь в кризисе."
)
print(f"Прогнозируемый рейтинг: {rating}")