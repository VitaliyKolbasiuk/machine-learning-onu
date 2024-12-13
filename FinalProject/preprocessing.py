import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def load_and_preprocess_data(data_path: str):
    df = pd.read_csv(data_path, delimiter=',')

    print(f"Кількість змінних до препроцессінгу: {df.shape[1]}")

    # Видаляємо стовпці з більш ніж 50% пропусків
    df = df.loc[:, df.isnull().sum() <= len(df) * 0.5]

    # Видаляємо стовпчики, які містять id, voice, calls, sms, cost
    df = df.drop(df.filter(regex='id|voice|calls|^sms|cost').columns, axis=1)

    print(f"Кількість змінних після препроцессінгу: {df.shape[1]}")

    y = df['target']
    X = df.drop('target', axis=1)

    # імпутація та масштабування
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    scaler = StandardScaler()
    pipe = Pipeline([('removenan', imputer), ('scale', scaler)])
    X = pipe.fit_transform(X)

    return X, y, df