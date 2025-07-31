import pandas as pd
import numpy as np
from .model import MovieRatingPredictor
from .nlp import PlotVectorizer

class Kinovanga:
    def __init__(self, model_path: str = None):
        if model_path:
            self.predictor = MovieRatingPredictor.load(model_path)
        else:
            self.predictor = None
        self.vectorizer = None
        self.director_avg = {}

    def train(self, df: pd.DataFrame):
        from .preprocessing import create_features
        from .nlp import PlotVectorizer

        df = create_features(df)
        self.director_avg = df.groupby('directors')['averageRating'].mean().to_dict()
        self.vectorizer = PlotVectorizer()

        # NLP
        desc = df['primaryTitle']  # В реальности — description, если есть
        X_text = self.vectorizer.fit_transform(desc)

        # Числовые + категориальные
        X_num = df[['startYear', 'runtimeMinutes', 'director_avg_rating', 'is_remake']].values
        X = np.hstack([X_num, X_text])

        y = df['averageRating'].values
        self.predictor = MovieRatingPredictor()
        self.predictor.fit(X, y)

    def predict_rating(self, title: str, director: str = "Unknown", year: int = 2020,
                       runtime: int = 120, description: str = ""):
        if not self.predictor or not self.vectorizer:
            raise RuntimeError("Model not trained. Call .train() first.")

        # Дефолтные значения
        year = year if pd.notna(year) else 2000
        runtime = runtime if pd.notna(runtime) else 90
        director = str(director) if pd.notna(director) else "Unknown"
        description = str(description) if description else title

        # Признаки
        is_remake = 1 if 'remake' in title.lower() else 0
        director_avg = self.director_avg.get(director, np.mean(list(self.director_avg.values())))

        # NLP
        text_vec = self.vectorizer.transform(pd.Series([description]))

        # Финальный вектор
        input_data = np.array([[year, runtime, director_avg, is_remake]])
        input_data = np.hstack([input_data, text_vec])

        rating = self.predictor.predict(input_data)[0]
        return round(max(1.0, min(10.0, rating)), 1)