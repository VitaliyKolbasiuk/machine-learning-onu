import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt

ALPHAS = np.logspace(-4, 4, 10)
RANDOM_STATE = 42
MAX_ITER = 5000

def detect_outliers(column):
    """Визначає викиди за допомогою методу міжквартильного розмаху (IQR)."""
    column = np.array(column)
    Q1 = np.percentile(column, 25)
    Q3 = np.percentile(column, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return np.where((column < lower_bound) | (column > upper_bound))[0]

def evaluate_model(model, X_test, y_test, model_name="Модель"):
    """Оцінює модель за метриками MSE, MAE, RMSE, та MAPE."""
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    print(f"{model_name} - MSE: {mse}, MAE: {mae}, MAPE: {mape}")
    return mse

# Завантаження набору даних
df = pd.read_csv("../data/OnlineNewsPopularityReduced.csv", delimiter=',')

# Вибір ознак і цільової змінної
X = df[['n_non_stop_unique_tokens', 'timedelta', 'n_tokens_title', 
        'average_token_length', 'LDA_03', 'num_imgs', 'num_videos']]
y = df['shares']

# Виявлення та видалення викидів
outlier_indices = np.unique(np.concatenate([
    detect_outliers(df['shares']),
    detect_outliers(df['average_token_length']),
    detect_outliers(df['n_non_stop_unique_tokens'])
]))
X_cleaned = X.drop(index=outlier_indices).reset_index(drop=True)
y_cleaned = y.drop(index=outlier_indices).reset_index(drop=True)

# Розділення набору даних
X_train, X_test, y_train, y_test = train_test_split(X_cleaned, y_cleaned, test_size=0.2, random_state=RANDOM_STATE)

# Лінійна регресія
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
print("Результати для Лінійної Регресії:")
evaluate_model(linear_model, X_test, y_test, "Лінійна Регресія")

# Ridge, Lasso, ElasticNet із налаштуванням гіперпараметрів
param_grid = {'alpha': ALPHAS}

ridge = GridSearchCV(Ridge(), param_grid, scoring='neg_mean_squared_error', n_jobs=-1)
lasso = GridSearchCV(Lasso(), param_grid, scoring='neg_mean_squared_error', n_jobs=-1)
elastic = GridSearchCV(ElasticNet(max_iter=MAX_ITER), {'alpha': ALPHAS, 'l1_ratio': [0.5]}, 
                       scoring='neg_mean_squared_error', n_jobs=-1)

# Тренування моделей
models = {"Ridge": ridge, "Lasso": lasso, "ElasticNet": elastic}
for name, model in models.items():
    print(f"Тренування моделі {name}...")
    model.fit(X_train, y_train)

# Оцінка та порівняння результатів
print("\nРезультати регуляризації:")
results = {}
for name, model in models.items():
    best_model = model.best_estimator_
    mse = evaluate_model(best_model, X_test, y_test, name)
    results[name] = {"model": best_model, "mse": mse}

# Побудова кривих валідації
plt.figure(figsize=(10, 6))
for name, model in models.items():
    plt.plot(ALPHAS, -model.cv_results_['mean_test_score'], label=name)
plt.xscale('log')
plt.xlabel('Регуляризація')
plt.ylabel('MSE')
plt.title('Криві валідації для Ridge, Lasso та ElasticNet')
plt.legend()
plt.grid()
plt.show()

# Коефіцієнти ElasticNet
elastic_model = results["ElasticNet"]["model"]
plt.figure(figsize=(10, 6))
plt.barh(X.columns, elastic_model.coef_, color='blue')
plt.title('Коефіцієнти ElasticNet')
plt.xlabel('Значення коефіцієнта')
plt.tight_layout()
plt.show()