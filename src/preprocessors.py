from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, LabelEncoder

class MissingValueHandler(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.2):
        self.threshold = threshold

    def fit(self, X, y=None):
        self.columns_to_drop_ = X.isnull().mean()[lambda x: x > self.threshold].index.tolist()
        return self

    def transform(self, X):
        X = X.drop(columns=self.columns_to_drop_)
        return X.dropna()

# 2. Binary mapper
class BinaryMapper(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None, mapping={'No': 0, 'Yes': 1}):
        self.columns = columns
        self.mapping = mapping

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.columns or []:
            if col in X.columns:
                X[col] = X[col].map(self.mapping)
        return X

# 3. Outlier remover
class OutlierRemover(BaseEstimator, TransformerMixin):
    def __init__(self, features=None, quantile=0.99):
        self.features = features
        self.quantile = quantile

    def fit(self, X, y=None):
        self.thresholds_ = {
            feature: X[feature].quantile(self.quantile)
            for feature in self.features or [] if feature in X.columns
        }
        return self

    def transform(self, X):
        X = X.copy()
        for feature, threshold in self.thresholds_.items():
            X = X[(X[feature] < threshold) | X[feature].isnull()]
        return X

# 4. Numerical scaler
class NumericalScaler(BaseEstimator, TransformerMixin):
    def __init__(self, exclude=None):
        self.exclude = exclude
        self.scaler = StandardScaler()

    def fit(self, X, y=None):
        self.numerical_features_ = X.select_dtypes(include='number').columns.tolist()
        if self.exclude in self.numerical_features_:
            self.numerical_features_.remove(self.exclude)
        self.scaler.fit(X[self.numerical_features_])
        return self

    def transform(self, X):
        X = X.copy()
        X[self.numerical_features_] = self.scaler.transform(X[self.numerical_features_])
        return X

# 5. Categorical encoder
class CategoricalEncoder(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.categorical_features_ = X.select_dtypes(include=['object', 'category']).columns.tolist()
        self.encoders_ = {
            col: LabelEncoder().fit(X[col].astype(str))
            for col in self.categorical_features_
        }
        return self

    def transform(self, X):
        X = X.copy()
        for col, encoder in self.encoders_.items():
            X[col] = encoder.transform(X[col].astype(str))
        return X
