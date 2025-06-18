import os
import joblib

import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline



def prepare_data(filepath="data/realty_data.csv"):

    estate = pd.read_csv(filepath)

    estate = estate.drop(columns=["product_name", "postcode", "address_name", "object_type", "period", "settlement", "area", "district", "description", "source"], axis=1)
    estate = estate.dropna()

    # Разделим признаки и целевую переменную
    x = estate.drop(columns="price")
    y = estate["price"]
    return x, y


def train_model(x, y):

    # Обработаем категориальные признаки
    cat_features = [col for col in x.columns if x[col].dtype == "object"]
    num_features = [col for col in x.columns if x[col].dtype != "object"]

    preprocessor = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features)
    ], remainder="passthrough")

    # Модель
    model = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=85)

    # Pipeline
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", model)
    ])

    # Обучение
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=85)
    pipeline.fit(x_train, y_train)

    # Сохраняем pipeline
    joblib.dump(pipeline, "model/model.pkl")


def read_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError("Model file not exists")

    model = joblib.load(model_path)
    return model