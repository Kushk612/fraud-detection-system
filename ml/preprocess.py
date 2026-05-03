import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class FraudPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.fitted = False

    def split_data(self, df):
        """
        Split features and target
        """
        X = df.drop("Class", axis=1)
        y = df["Class"]

        return X, y

    def train_test_split_data(self, X, y):
        """
        Split dataset into train/test
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y  # IMPORTANT for imbalance
        )

        return X_train, X_test, y_train, y_test

    def fit_transform(self, X_train):
        """
        Fit scaler ONLY on training data (avoid leakage)
        """
        X_train_scaled = self.scaler.fit_transform(X_train)
        self.fitted = True
        return X_train_scaled

    def transform(self, X_test):
        """
        Transform test data using same scaler
        """
        if not self.fitted:
            raise Exception("Scaler not fitted. Call fit_transform first.")

        return self.scaler.transform(X_test)

    def preprocess(self, df):
        """
        Full pipeline in one call
        """
        X, y = self.split_data(df)
        X_train, X_test, y_train, y_test = self.train_test_split_data(X, y)

        X_train_scaled = self.fit_transform(X_train)
        X_test_scaled = self.transform(X_test)

        return X_train_scaled, X_test_scaled, y_train, y_test