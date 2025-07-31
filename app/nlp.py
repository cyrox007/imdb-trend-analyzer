from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np

class PlotVectorizer:
    def __init__(self, max_features=100):
        self.tfidf = TfidfVectorizer(max_features=max_features, stop_words='english')
        self.is_fitted = False

    def fit_transform(self, texts):
        X = self.tfidf.fit_transform(texts.fillna('')).toarray()
        self.is_fitted = True
        return X

    def transform(self, texts):
        if not self.is_fitted:
            raise RuntimeError("Vectorizer not fitted. Call fit_transform first.")
        X = self.tfidf.transform(texts.fillna('')).toarray()
        return X

    def get_feature_names(self):
        return [f"tfidf_{name}" for name in self.tfidf.get_feature_names_out()]