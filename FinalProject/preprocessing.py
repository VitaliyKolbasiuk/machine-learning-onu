import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def load_and_preprocess_data(data_path: str):
    df = pd.read_csv(data_path, delimiter=',')
    print(f"Кількість змінних до препроцессінгу: {df.shape[1]}")
    # Видаляємо стовпці з більш ніж 75% пропусків
    important_columns = ['target', 'important_feature']
    df = df.loc[:, (df.isnull().sum() <= len(df) * 0.75) | df.columns.isin(important_columns)]
    # Видаляємо стовпчики, які містять id, voice, calls, sms, count_url_category_
    df = df.drop(df.filter(regex='^voice|id|^sms|^count_url_category_|calls').columns, axis=1)
    print(f"Кількість змінних після препроцессінгу: {df.shape[1]}")
    y = df['target']
    X = df.drop('target', axis=1)
    # Імпутація за допомогою IterativeImputer та масштабування
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    scaler = StandardScaler()
    pipe = Pipeline([('removenan', imputer), ('scale', scaler)])
    X = pipe.fit_transform(X)
    return X, y, df