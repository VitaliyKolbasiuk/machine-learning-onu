import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def load_and_preprocess_data(data_path: str):
    # Завантаження даних
    df = pd.read_csv(data_path, delimiter=',')
    print(f"Кількість змінних до препроцессінгу: {df.shape[1]}")

    # Видаляємо стовпці з більш ніж 75% пропусків
    important_columns = ['target', 'important_feature']
    df = df.loc[:, (df.isnull().sum() <= len(df) * 0.75) | df.columns.isin(important_columns)]

    # Видаляємо стовпчики, які містять id, voice, calls, sms, count_url_category_
    df = df.drop(df.filter(regex='^voice|id|^sms|^count_url_category_|calls').columns, axis=1)
    print(f"Кількість змінних після препроцессінгу: {df.shape[1]}")

    # Розділення на цільову змінну та ознаки
    y = df['target']
    X = df.drop('target', axis=1)

    # Визначення числових і категоріальних змінних
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object', 'category']).columns

    # Пайплайн для числових змінних
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    # Пайплайн для категоріальних змінних
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Об’єднання трансформаторів у ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )

    X = preprocessor.fit_transform(X)

    return X, y, df