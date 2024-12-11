import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import (GridSearchCV, KFold, train_test_split,
                                     validation_curve)
from sklearn.tree import DecisionTreeRegressor, plot_tree

plt.figure(figsize=(10, 6))


def detect_outliers(column):
    col_array = np.array(column)
    Q1 = np.percentile(col_array, 25)
    Q3 = np.percentile(col_array, 75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    return np.where((col_array < lower_bound) | (col_array > upper_bound))[0]


def plot_validation_curves(estimator, X, y, param_name, param_range, cv, scoring='neg_mean_squared_error'):
    train_scores, test_scores = validation_curve(
        estimator,
        X, y,
        param_name=param_name,
        param_range=param_range,
        scoring=scoring,
        cv=cv,
        n_jobs=-1
    )

    train_scores_mean = -train_scores.mean(axis=1)
    test_scores_mean = -test_scores.mean(axis=1)

    plt.plot(param_range, train_scores_mean, label='Тренувальний MSE', color='blue', marker='o')
    plt.plot(param_range, test_scores_mean, label='Тестовий MSE', color='orange', marker='o')
    plt.xlabel(param_name)
    plt.ylabel('Середньоквадратична помилка')
    plt.title(f'Крива валідації для {param_name}')
    plt.legend()
    plt.grid()
    plt.show()


def main():
    df = pd.read_csv("../data/OnlineNewsPopularityReduced.csv", delimiter=',')

    X_full = df[['n_non_stop_unique_tokens', 'timedelta', 'n_tokens_title',
                 'average_token_length', 'LDA_03', 'num_imgs', 'num_videos']]
    y_full = df['shares']

    outliers_shares = detect_outliers(df['shares'])
    outliers_avgtoken = detect_outliers(df['average_token_length'])
    outliers_nonstop = detect_outliers(df['n_non_stop_unique_tokens'])

    combined_outliers = np.unique(
        np.concatenate((outliers_shares, outliers_avgtoken, outliers_nonstop))
    )

    X = X_full.drop(index=combined_outliers).reset_index(drop=True)
    y = y_full.drop(index=combined_outliers).reset_index(drop=True)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 1. Decision Tree з базовими параметрами
    regressor = DecisionTreeRegressor(random_state=42)
    regressor.fit(X_train, y_train)
    y_pred = regressor.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Середньоквадратична помилка (MSE): {mse:.2f}")

    # 2. Крос-валідація та підбір найкращих гіперпараметрів для DecisionTreeRegressor
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    param_grid_dt = {
        'max_depth': [3, 5, 10, 15, 20],
        'min_samples_split': [2, 5, 10, 20],
        'min_samples_leaf': [1, 2, 5, 10],
        'max_features': ['sqrt', 'log2']
    }

    grid_search = GridSearchCV(
        estimator=DecisionTreeRegressor(random_state=42),
        param_grid=param_grid_dt,
        scoring='neg_mean_squared_error',
        cv=kf,
        n_jobs=-1,
        verbose=0
    )
    grid_search.fit(X, y)

    best_params = grid_search.best_params_
    best_mse = -grid_search.best_score_

    print("Найкращі параметри для моделі:")
    print(best_params)
    print(f"MSE за результатами крос-валідації: {best_mse:.2f}")

    # Криві валідації для DecisionTreeRegressor
    parameters_dt = {
        'max_depth': [3, 5, 10, 15, 20],
        'min_samples_split': [2, 5, 10, 20],
        'min_samples_leaf': [1, 2, 5, 10]
    }

    for param_name, param_range in parameters_dt.items():
        plot_validation_curves(DecisionTreeRegressor(random_state=42), X, y, param_name, param_range, cv=kf)

    # Найкраща модель DecisionTreeRegressor
    best_regressor = DecisionTreeRegressor(
        max_depth=3,
        max_features=None,
        min_samples_leaf=5,
        min_samples_split=2,
        random_state=42
    )
    best_regressor.fit(X, y)

    plt.figure(figsize=(16, 10))
    plot_tree(best_regressor, feature_names=X.columns.tolist(), filled=True, fontsize=10)
    plt.title("Візуалізація дерева рішень")
    plt.show()

    # Важливість ознак для дерева рішень
    feature_importances = best_regressor.feature_importances_
    sns.barplot(x=feature_importances, y=X.columns)
    plt.title('Важливість ознак')
    plt.xlabel('Важливість')
    plt.ylabel('Ознака')
    plt.grid()
    plt.show()

    # 3. RandomForestRegressor з параметрами за замовчуванням
    rf_default = RandomForestRegressor(random_state=42)
    rf_default.fit(X_train, y_train)
    y_pred_rf = rf_default.predict(X_test)
    mse_rf = mean_squared_error(y_test, y_pred_rf)
    print(f"Середньоквадратична помилка (MSE) для Random Forest (параметри за замовчуванням): {mse_rf:.2f}")

    # Підбір гіперпараметрів для RandomForestRegressor
    param_grid_rf = {
        'n_estimators': [50, 100, 200],
        'max_depth': [10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2']
    }

    rf_grid = GridSearchCV(
        RandomForestRegressor(random_state=42),
        param_grid_rf,
        scoring='neg_mean_squared_error',
        cv=kf,
        n_jobs=-1,
        verbose=0
    )
    rf_grid.fit(X_train, y_train)

    best_rf = rf_grid.best_estimator_
    y_pred_rf_best = best_rf.predict(X_test)
    mse_rf_best = mean_squared_error(y_test, y_pred_rf_best)

    print(f"Найкращі параметри для Random Forest: {rf_grid.best_params_}")
    print(f"Середньоквадратична помилка (MSE) для Random Forest (оптимізована): {mse_rf_best:.2f}")

    # Криві валідації для RandomForestRegressor
    parameters_rf = {
        'n_estimators': [50, 100, 200],
        'max_depth': [10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2']
    }

    for param_name, param_range in parameters_rf.items():
        plot_validation_curves(RandomForestRegressor(random_state=42), X, y, param_name, param_range, cv=kf)

    # Важливість ознак для RandomForestRegressor
    feature_importances_rf = pd.DataFrame({
        'Feature': X.columns,
        'Importance': best_rf.feature_importances_
    }).sort_values(by='Importance', ascending=False)

    top_features = feature_importances_rf.head(10)
    sns.barplot(data=top_features, x='Importance', y='Feature')
    plt.title('Топ-10 важливих ознак (Random Forest)')
    plt.xlabel('Важливість')
    plt.ylabel('Ознака')
    plt.grid()
    plt.show()


if __name__ == "__main__":
    main()
