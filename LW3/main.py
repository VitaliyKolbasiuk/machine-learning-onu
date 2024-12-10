import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split, KFold, GridSearchCV, cross_val_score
from sklearn.neighbors import KNeighborsRegressor, RadiusNeighborsRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

def detect_outliers(data):
    Q1, Q3 = np.percentile(data, [25, 75])
    IQR = Q3 - Q1
    lower_bound, upper_bound = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    return np.where((data < lower_bound) | (data > upper_bound))[0]

df = pd.read_csv("../data/OnlineNewsPopularityReduced.csv", delimiter=',')
features = ['n_non_stop_unique_tokens', 'timedelta', 'n_tokens_title', 
            'average_token_length', 'LDA_03', 'num_imgs', 'num_videos']
X, y = df[features], df['shares']

# Об'єднання індексів викидів
outlier_indices = np.unique(np.concatenate([
    detect_outliers(df[col]) for col in ['shares', 'average_token_length', 'n_non_stop_unique_tokens']
]))

# Видалення викидів
X, y = X.drop(index=outlier_indices).reset_index(drop=True), y.drop(index=outlier_indices).reset_index(drop=True)

# Розподіл на навчальну та тестову вибірки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Масштабування даних
scaler = RobustScaler().fit(X_train)
X_train_scaled, X_test_scaled = scaler.transform(X_train), scaler.transform(X_test)

# Навчання та оцінка моделі KNeighborsRegressor
knn_regressor = KNeighborsRegressor(n_neighbors=5).fit(X_train_scaled, y_train)
y_pred = knn_regressor.predict(X_test_scaled)
mse, mae = mean_squared_error(y_test, y_pred), mean_absolute_error(y_test, y_pred)

print(f'Середньоквадратична помилка (MSE): {mse}')
print(f'Середня абсолютна помилка (MAE): {mae}')

# Статистичні характеристики тестової вибірки
print(f'Середнє значення shares: {y_test.mean()}')
print(f'Медіана shares: {y_test.median()}')
print(f'Стандартне відхилення shares: {y_test.std()}')

# Налаштування параметра k через GridSearchCV
kf = KFold(n_splits=5, shuffle=True, random_state=42)
param_grid = {'n_neighbors': list(range(1, 51))}
grid_search = GridSearchCV(KNeighborsRegressor(), param_grid, cv=kf, scoring='neg_mean_absolute_error')
grid_search.fit(X_train_scaled, y_train)

best_k, best_mae = grid_search.best_params_['n_neighbors'], -grid_search.best_score_
print(f'Найкраще значення k: {best_k}')
print(f'Найкраще значення MAE: {best_mae}')

plt.figure(figsize=(10, 6))
plt.plot(param_grid['n_neighbors'], -grid_search.cv_results_['mean_test_score'], marker='o')
plt.xlabel('Кількість сусідів (k)')
plt.ylabel('Середня абсолютна помилка (MAE)')
plt.title('Залежність MAE від числа сусідів (k)')
plt.grid()
plt.show()

# Оптимізація параметра p
best_p, best_score = max(
    ((p, cross_val_score(KNeighborsRegressor(n_neighbors=best_k, metric='minkowski', p=p, weights='distance'), X, y, cv=5, scoring='neg_mean_absolute_error').mean())
     for p in np.linspace(1, 10, 20)),
    key=lambda x: x[1]
)
print(f"Оптимальне значення параметра p: {best_p}")
print(f"Найкраще середнє значення MAE: {-best_score}")

param_grid = {'radius': np.linspace(0.1, 10, 50)}
radius_grid_search = GridSearchCV(
    RadiusNeighborsRegressor(weights='distance', metric='minkowski', p=2),
    param_grid, cv=5, scoring='neg_mean_absolute_error'
)
radius_grid_search.fit(X_train_scaled, y_train)

best_radius = radius_grid_search.best_params_['radius']
print(f"Оптимальний радіус: {best_radius}")

# Оцінка моделі з оптимальним радіусом
radius_regressor = RadiusNeighborsRegressor(radius=best_radius, weights='distance', metric='minkowski', p=2)
radius_regressor.fit(X_train_scaled, y_train)
y_pred_radius = radius_regressor.predict(X_test_scaled)

mse_radius = mean_squared_error(y_test, y_pred_radius)
mae_radius = mean_absolute_error(y_test, y_pred_radius)

print(f"Середньоквадратична помилка (MSE) для RadiusNeighborsRegressor: {mse_radius}")
print(f"Середня абсолютна помилка (MAE) для RadiusNeighborsRegressor: {mae_radius}")