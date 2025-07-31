# kinovanga/predictor.py
import pandas as pd
import numpy as np
import joblib
import gc
from tqdm import tqdm

# Внешние импорты вверху
from app.data_loader import load_imdb_chunked
from app.preprocessing import create_features
from app.nlp import PlotVectorizer
from app.model import MovieRatingPredictor
from utils.counter import count_lines_gz


class Kinovanga:
    def __init__(self, model_path: str = None):
        if model_path:
            data = joblib.load(model_path)
            self.predictor = data['model']
            self.vectorizer = data['vectorizer']
            self.director_avg = data['director_avg']
        else:
            self.predictor = None
            self.vectorizer = None
            self.director_avg = {}

    def train_on_chunks(
        self,
        basics_path: str,
        ratings_path: str,
        crew_path: str,
        chunksize: int = 10000,
        max_chunks: int = None,
        val_split: float = 0.2
    ):
        print("🚀 Начинаем обучение КиноВанги на чанках...")

        total_lines = count_lines_gz(basics_path) - 1  # минус заголовок
        total_chunks = (total_lines + chunksize - 1) // chunksize

        print(f"  • Размер чанка: {chunksize}")
        print(f"  • Всего данных: ~{total_lines:,} строк")
        print(f"  • Оценка чанков: ~{total_chunks}")
        print(f"  • Максимум чанков: {max_chunks if max_chunks else f'все ({total_chunks})'}")
        print(f"  • Валидация: {val_split * 100:.0f}%")

        # --- 1. Подготовка векторизатора ---
        print("🧠 Обучаю TF-IDF векторизатор на первом чанке...")
        chunk_iter = load_imdb_chunked(basics_path, ratings_path, crew_path, chunksize)
        
        # Только для TF-IDF
        first_chunk = next(chunk_iter)
        first_chunk = create_features(first_chunk)
        
        if 'description' not in first_chunk.columns:
            raise KeyError("Колонка 'description' отсутствует. Проверьте create_features().")
        
        self.vectorizer = PlotVectorizer(max_features=100)
        self.vectorizer.fit_transform(first_chunk['description'])

        # --- 2. Сбор статистики по режиссёрам и обучение ---
        print("📊 Собираем статистику и обучаем модель...")
        director_ratings = {}  # {director: [ratings]}

        def update_director_stats(df):
            for _, row in df[['directors', 'averageRating']].iterrows():
                director = row['directors']
                rating = row['averageRating']
                if pd.notna(director) and pd.notna(rating):
                    if director not in director_ratings:
                        director_ratings[director] = []
                    director_ratings[director].append(rating)

        # Инициализация модели
        self.predictor = MovieRatingPredictor()
        val_history = []
        total_chunks = 0

        # Возвращаем первый чанк в поток обработки
        chunks = [first_chunk] + list(chunk_iter)

        for chunk in chunks:
            if max_chunks and total_chunks >= max_chunks:
                break

            # Обновляем статистику ПЕРЕД тем, как использовать director_avg
            update_director_stats(chunk)

            # Создаём director_avg_map на основе текущей статистики
            director_avg_map = {k: np.mean(v) for k, v in director_ratings.items()}

            # Один раз — создаём признаки с актуальной статистикой
            df = create_features(chunk, director_avg_map=director_avg_map)

            # Пропускаем строки без рейтинга
            df = df.dropna(subset=['averageRating'])

            X_text = self.vectorizer.transform(df['description'])
            X_num = df[['startYear', 'runtimeMinutes', 'director_avg_rating', 'is_remake']].values
            X = np.hstack([X_num, X_text.toarray()])  # или использовать sparse
            X = np.nan_to_num(X, nan=0.0)
            y = df['averageRating'].values

            # Валидация
            split_idx = int(len(X) * (1 - val_split))
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]

            self.predictor.partial_fit(X_train, y_train)

            y_pred = self.predictor.predict(X_val)
            mae = np.mean(np.abs(y_val - y_pred))
            val_history.append(mae)

            tqdm.write(f"Chunk {total_chunks + 1:2d} | MAE: {mae:.3f}")
            total_chunks += 1

            del df, X, y, X_train, X_val, y_train, y_val
            gc.collect()

        # Сохраняем финальные средние
        self.director_avg = {k: np.mean(v) for k, v in director_ratings.items()}
        print("✅ Обучение завершено.")
        print(f"📊 Финальный MAE: {np.mean(val_history[-3:]):.3f}")
        
    def predict_rating(
        self,
        title: str,
        director: str = "Unknown",
        year: int = 2020,
        runtime: int = 120,
        description: str = ""
    ) -> float:
        """
        Предсказывает рейтинг фильма.
        """
        if not self.predictor or not self.vectorizer:
            raise RuntimeError("Model not trained. Call .train_on_chunks() first.")

        # Дефолтные значения
        year = year if pd.notna(year) else 2000
        runtime = runtime if pd.notna(runtime) else 90
        director = str(director) if pd.notna(director) else "Unknown"
        description = str(description) if description else str(title)

        # Признаки
        is_remake = 1 if 'remake' in title.lower() else 0
        director_avg = self.director_avg.get(director, np.mean(list(self.director_avg.values())))

        # NLP
        text_vec = self.vectorizer.transform(pd.Series([description]))

        # Финальный вектор
        X_num = np.array([[year, runtime, director_avg, is_remake]])
        X = np.hstack([X_num, text_vec])

        rating = self.predictor.predict(X)[0]
        return round(max(1.0, min(10.0, rating)), 1)

    def save(self, path: str):
        """Сохраняет всю модель целиком."""
        joblib.dump({
            'model': self.predictor,
            'vectorizer': self.vectorizer,
            'director_avg': self.director_avg
        }, path)