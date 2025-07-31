from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
import joblib

class MovieRatingPredictor:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        self.imputer = SimpleImputer(strategy='median')
        self.is_fitted = False

    def fit(self, X, y):
        X = self.imputer.fit_transform(X)
        self.model.fit(X, y)
        self.is_fitted = True

    def predict(self, X):
        if not self.is_fitted:
            raise RuntimeError("Model not fitted.")
        X = self.imputer.transform(X)
        return self.model.predict(X)

    def save(self, path: str):
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str):
        return joblib.load(path)
    