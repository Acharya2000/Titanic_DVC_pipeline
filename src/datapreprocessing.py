from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import pandas as pd 
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import os
import numpy as np
import joblib

class Preprocessing:
    def __init__(self, path_data, test_size=0.2, random_state=42, processed_dir="data/processed"):
        self.test_size = test_size
        self.random_state = random_state
        self.processed_dir = processed_dir

        # Load dataset
        self.df = pd.read_csv(path_data)

        # Drop irrelevant columns
        self.df = self.df.drop(
            ["PassengerId", "Pclass", "Name", "Parch", "Ticket", "Cabin", "Embarked"],
            axis=1
        )

        # Separate target and features
        self.y = self.df["Survived"]
        self.x = self.df.drop("Survived", axis=1)

        # Identify numerical & categorical features
        self.num_features = self.x.select_dtypes(include=["int64", "float64"]).columns
        self.cat_features = self.x.select_dtypes(include=["object"]).columns

        # Define pipeline for numerical columns
        num_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])

        # Define pipeline for categorical columns
        cat_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ])

        # Final column transformer
        self.cl = ColumnTransformer(
            transformers=[
                ("num", num_transformer, self.num_features),
                ("cat", cat_transformer, self.cat_features)
            ]
        )

        # Ensure processed dir exists
        os.makedirs(self.processed_dir, exist_ok=True)

    def preprocessing_data(self):
        """Split data and preprocess"""
        x_train, x_test, y_train, y_test = train_test_split(
            self.x, self.y, test_size=self.test_size, random_state=self.random_state
        )

        x_train_preprocessed = self.cl.fit_transform(x_train)
        x_test_preprocessed = self.cl.transform(x_test)

        return x_train_preprocessed, x_test_preprocessed, y_train, y_test
    
    def save_preprocessed_data(self):
        """Save preprocessed arrays and pipeline"""
        X_train, X_test, y_train, y_test = self.preprocessing_data()

        np.save(os.path.join(self.processed_dir, "X_train.npy"), X_train)
        np.save(os.path.join(self.processed_dir, "X_test.npy"), X_test)
        np.save(os.path.join(self.processed_dir, "y_train.npy"), y_train)
        np.save(os.path.join(self.processed_dir, "y_test.npy"), y_test)

        joblib.dump(self.cl, os.path.join(self.processed_dir, "preprocessor.pkl"))

        print(f"[INFO] Processed data & pipeline saved in {self.processed_dir}")
