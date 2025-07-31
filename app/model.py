from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import joblib
import numpy as np

class MovieRatingPredictor:
    def __init__(self):
        # Ключевые параметры для стабильности
        self.model = SGDRegressor(
            loss='squared_error',
            learning_rate='adaptive',
            eta0=0.001,           # Меньше скорость обучения
            random_state=42,
            max_iter=100,
            tol=1e-4,
            early_stopping=True,  # Ранняя остановка
            validation_fraction=0.1,
            n_iter_no_change=5
        )
        self.imputer = SimpleImputer(strategy='median')
        self.scaler = StandardScaler()
        self.is_fitted = False

    def partial_fit(self, X, y):
        # Убедимся, что X и y — числовые и без NaN
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32).ravel()

        # Заменяем NaN и inf
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        y = np.nan_to_num(y, nan=np.mean(y), posinf=np.median(y), neginf=np.median(y))

        # Импутация
        X = self.imputer.fit_transform(X) if not self.is_fitted else self.imputer.transform(X)

        # Масштабирование
        X = self.scaler.fit_transform(X) if not self.is_fitted else self.scaler.transform(X)

        # Обучение
        if not self.is_fitted:
            self.model.fit(X, y)
            self.is_fitted = True
        else:
            self.model.partial_fit(X, y)

    def predict(self, X):
        if not self.is_fitted:
            raise RuntimeError("Model not trained.")

        X = np.asarray(X, dtype=np.float32)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        X = self.imputer.transform(X)
        X = self.scaler.transform(X)

        pred = self.model.predict(X)
        return np.clip(pred, 1.0, 10.0)  # Ограничиваем от 1 до 10

    def save(self, path: str):
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str):
        return joblib.load(path)