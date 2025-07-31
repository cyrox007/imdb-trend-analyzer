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
        """
        Обучение модели на чанках с валидацией и прогресс-баром.
        """
        print("🚀 Начинаем обучение КиноВанги на чанках...")
        print(f"  • Размер чанка: {chunksize}")
        print(f"  • Максимум чанков: {max_chunks if max_chunks else 'все'}")
        print(f"  • Валидация: {val_split * 100:.0f}%")

        # --- 1. Подготовка векторизатора ---
        print("🧠 Обучаю TF-IDF векторизатор на первом чанке...")
        chunk_iter = load_imdb_chunked(basics_path, ratings_path, crew_path, chunksize)
        df_chunk = next(chunk_iter)
        df_chunk = create_features(df_chunk)

        if 'description' not in df_chunk.columns:
            raise KeyError("Колонка 'description' отсутствует. Проверьте create_features().")

        self.vectorizer = PlotVectorizer(max_features=100)
        self.vectorizer.fit_transform(df_chunk['description'])

        # --- 2. Сбор статистики по режиссёрам ---
        print("📊 Собираем статистику по режиссёрам...")
        all_director_data = []
        chunk_iter = load_imdb_chunked(basics_path, ratings_path, crew_path, chunksize)
        for i, chunk in enumerate(chunk_iter):
            processed = create_features(chunk)  # Просто для извлечения рейтингов
            all_director_data.append(processed[['directors', 'averageRating']])
            if max_chunks and i >= max_chunks:
                break
        full_df = pd.concat(all_director_data, ignore_index=True)
        self.director_avg = full_df.groupby('directors')['averageRating'].mean().to_dict()

        # --- 3. Инициализация модели ---
        self.predictor = MovieRatingPredictor()
        val_history = []

        # --- 4. Обучение по чанкам ---
        print("🏋️ Обучаю модель...")
        chunk_iter = load_imdb_chunked(basics_path, ratings_path, crew_path, chunksize)
        for i, chunk in enumerate(tqdm(chunk_iter, desc="📦 Чанки", total=max_chunks)):
            if max_chunks and i >= max_chunks:
                break

            # 🔥 Обработка с передачей director_avg
            df = create_features(chunk, director_avg_map=self.director_avg)

            # Теперь director_avg_rating точно есть
            X_text = self.vectorizer.transform(df['description'])
            X_num = df[['startYear', 'runtimeMinutes', 'director_avg_rating', 'is_remake']].values
            X = np.hstack([X_num, X_text])
            X = np.nan_to_num(X, nan=0.0)
            y = df['averageRating'].values

            # Валидация
            split_idx = int(len(X) * (1 - val_split))
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]

            self.predictor.partial_fit(X_train, y_train)

            # Оценка
            y_pred = self.predictor.predict(X_val)
            mae = np.mean(np.abs(y_val - y_pred))
            val_history.append(mae)

            tqdm.write(f"Chunk {i+1:2d} | MAE: {mae:.3f}")

            del df, X, y, X_train, X_val, y_train, y_val
            import gc; gc.collect()

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