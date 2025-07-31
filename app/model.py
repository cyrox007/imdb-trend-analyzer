from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import joblib
import numpy as np

class MovieRatingPredictor:
    def __init__(self):
        self.model = SGDRegressor(
            loss='squared_error',
            learning_rate='adaptive',
            eta0=0.01,
            random_state=42,
            max_iter=100,
            tol=1e-3
        )
        self.imputer = SimpleImputer(strategy='median')
        self.scaler = StandardScaler()  # Новый: масштабирование
        self.is_fitted = False

    def partial_fit(self, X, y):
        X = self.imputer.fit_transform(X) if not self.is_fitted else self.imputer.transform(X)
        X = self.scaler.fit_transform(X) if not self.is_fitted else self.scaler.transform(X)  # Масштабируем
        y = np.asarray(y).ravel()

        if not self.is_fitted:
            self.model.fit(X, y)
            self.is_fitted = True
        else:
            self.model.partial_fit(X, y)

    def predict(self, X):
        if not self.is_fitted:
            raise RuntimeError("Model not trained.")
        X = self.imputer.transform(X)
        X = self.scaler.transform(X)  # Масштабируем при предсказании
        return self.model.predict(X)

    def save(self, path: str):
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str):
        return joblib.load(path)
    