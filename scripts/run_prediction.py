# scripts/run_prediction.py
from app.predictor import Kinovanga
from app.data_loader import load_imdb_chunked
from app.preprocessing import create_features
from utils.counter import count_lines_gz
from tabulate import tabulate
import pandas as pd
import numpy as np
import joblib
import gc

# Пути к данным
basics_path = 'data/title.basics.tsv.gz'
ratings_path = 'data/title.ratings.tsv.gz'
crew_path = 'data/title.crew.tsv.gz'
principals_path = 'data/title.principals.tsv.gz'  # ✅ Используется

# === 1. Обучение модели ===
print("🎬 Запуск обучения КиноВанги с учётом жанров и актёров...")

kino = Kinovanga()

kino.train_on_chunks(
    basics_path=basics_path,
    ratings_path=ratings_path,
    crew_path=crew_path,
    principals_path=principals_path,
    chunksize=10000,
    max_chunks=None,
    val_split=0.2
)

# Сохраняем модель
kino.save("models/kinovanga_v2.pkl")
print("✅ Модель сохранена: models/kinovanga_v2.pkl")

# === 2. Предсказание для новых фильмов ===
new_movies = [
    {
        'title': "The Batman: Part II",
        'director': "Мэтт Ривз",
        'year': 2026,
        'runtime': 150,
        'description': "Бэтмен против Смерти. Темный рыцарь в кризисе.",
        'genres': "Action, Crime, Drama",
        'actors': ["Robert Pattinson", "Zoë Kravitz", "Paul Dano"]
    },
    {
        'title': "Дюна: Часть Третья",
        'director': "Дени Вильнёв",
        'year': 2026,
        'runtime': 165,
        'description': "Завершение эпической саги по романам Фрэнка Герберта",
        'genres': "Sci-Fi, Adventure, Drama",
        'actors': ["Timothée Chalamet", "Zendaya", "Rebecca Ferguson"]
    },
    {
        'title': "Avatar 3",
        'director': "Джеймс Кэмерон",
        'year': 2026,
        'runtime': 170,
        'description': "Продолжение эпопеи о Пандоре. В этом фильме будут больше акцентов на водной среде и новых племенах.",
        'genres': "Sci-Fi, Adventure, Action",
        'actors': ["Sam Worthington", "Zoe Saldana", "Sigourney Weaver"]
    }
]

# Предсказания
predictions = []
print("\n" + "="*60)
print("           🎬 ПРЕДСКАЗАНИЯ ДЛЯ НОВЫХ ФИЛЬМОВ")
print("="*60)

for movie in new_movies:
    try:
        rating = kino.predict_rating(
            title=movie['title'],
            director=movie.get('director', 'Unknown'),
            year=movie.get('year', 2000),
            runtime=movie.get('runtime', 90),
            description=movie.get('description', ''),
            genres=movie.get('genres', 'Unknown'),
            actors=movie.get('actors', [])
        )
        predictions.append({
            'Название': movie['title'],
            'Год': movie['year'],
            'Режиссёр': movie['director'],
            'Длительность': f"{movie['runtime']} мин",
            'Жанры': movie['genres'],
            'Актёры': ", ".join(movie['actors']) if movie['actors'] else "Неизвестны",
            'Рейтинг': f"⭐ {round(rating, 1)}"
        })
    except Exception as e:
        print(f"Ошибка для {movie['title']}: {e}")

# Вывод в терминале
print(tabulate(predictions, headers="keys", tablefmt="fancy_grid", stralign="center"))

# Сохранение в HTML (как в Colab)
df_pred = pd.DataFrame(predictions)
df_pred.to_html("predictions.html", index=False, escape=False)
print("\n✅ Результаты сохранены в predictions.html (открой в браузере)")

# === 3. Оценка на случайных примерах (опционально) ===
print("\n💡 Совет: создай `notebook.ipynb`, чтобы использовать `display()` и `.style.background_gradient()`")