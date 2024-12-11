import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, validation_curve
from sklearn.tree import DecisionTreeRegressor


def detect_outliers(column):
    column = np.array(column)
    Q1 = np.percentile(column, 25)
    Q3 = np.percentile(column, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers_indices = np.where((column < lower_bound) | (column > upper_bound))[0]
    return outliers_indices


df = pd.read_csv("../data/OnlineNewsPopularityReduced.csv", delimiter=',')

X_1 = df[['n_non_stop_unique_tokens', 'timedelta', 'n_tokens_title', 'average_token_length', 'LDA_03', 'num_imgs', 'num_videos']]
y_1 = df['shares']

outlier_indices_shares = detect_outliers(df['shares'])
outlier_indices_average_token_length = detect_outliers(df['average_token_length'])
outlier_indices_n_non_stop_unique_tokens = detect_outliers(df['n_non_stop_unique_tokens'])

combined_outlier_indices = np.unique(np.concatenate((outlier_indices_shares,
                                                     outlier_indices_average_token_length,
                                                     outlier_indices_n_non_stop_unique_tokens)))

X = X_1.drop(index=combined_outlier_indices).reset_index(drop=True)
y = y_1.drop(index=combined_outlier_indices).reset_index(drop=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

plt.figure(figsize=(10, 6))


# Допоміжна функція для побудови кривих валідації
def plot_validation_curves(model, X_train, y_train, param_name, param_range, title, xscale='linear'):
    train_scores, test_scores = validation_curve(
        model, X_train, y_train,
        param_name=param_name, param_range=param_range,
        scoring="neg_mean_squared_error", cv=5
    )
    train_mean = -np.mean(train_scores, axis=1)
    test_mean = -np.mean(test_scores, axis=1)

    plt.plot(param_range, train_mean, label="Оцінка на тренувальному наборі", marker='o')
    plt.plot(param_range, test_mean, label="Оцінка на тестовому наборі", marker='o')
    plt.title(title)
    plt.xlabel(param_name)
    plt.ylabel("Середньоквадратична помилка")
    if xscale == 'log':
        plt.xscale('log')
    plt.legend()
    plt.grid()


# Перевірка з AdaBoost
base_model = DecisionTreeRegressor(max_depth=4, random_state=42)

n_estimators_range = np.arange(10, 210, 20)
learning_rate_range = np.logspace(-2, 0.5, 10)

ada_boost = AdaBoostRegressor(estimator=base_model, random_state=42)
ada_boost.fit(X_train, y_train)
y_test_pred = ada_boost.predict(X_test)
mse = mean_squared_error(y_test, y_test_pred)
print(f"MSE для AdaBoost: {mse}")

plt.figure(figsize=(10, 6))

plt.subplot(1, 2, 1)
plot_validation_curves(ada_boost, X_train, y_train, "n_estimators", n_estimators_range,
                       "Криві валідації для n_estimators")

plt.subplot(1, 2, 2)
plot_validation_curves(ada_boost, X_train, y_train, "learning_rate", learning_rate_range,
                       "Криві валідації для learning_rate", xscale='log')

plt.tight_layout()
plt.show()

# Перевірка з GradientBoostingRegressor
gradient_boost = GradientBoostingRegressor(
    n_estimators=100, learning_rate=0.1, max_depth=4, random_state=42
)

gradient_boost.fit(X_train, y_train)
y_test_pred = gradient_boost.predict(X_test)
mse = mean_squared_error(y_test, y_test_pred)
print(f"MSE для Gradient Boosting: {mse}")

plt.figure(figsize=(10, 6))

plt.subplot(1, 2, 1)
plot_validation_curves(gradient_boost, X_train, y_train, "n_estimators", n_estimators_range,
                       "Криві валідації для n_estimators (Gradient Boosting)")

plt.subplot(1, 2, 2)
plot_validation_curves(gradient_boost, X_train, y_train, "learning_rate", learning_rate_range,
                       "Криві валідації для learning_rate (Gradient Boosting)", xscale='log')

plt.tight_layout()
plt.show()

# Реалізація XGBoost
xgboost_model = xgb.XGBRegressor(
    objective="reg:squarederror", random_state=42, eval_metric="rmse"
)

plt.figure(figsize=(15, 10))

plt.subplot(2, 2, 1)
plot_validation_curves(xgboost_model, X_train, y_train, "n_estimators", n_estimators_range,
                       "XGBoost: Криві валідації для n_estimators")

plt.subplot(2, 2, 2)
plot_validation_curves(xgboost_model, X_train, y_train, "learning_rate", learning_rate_range,
                       "XGBoost: Криві валідації для learning_rate", xscale='log')


# Реалізація LightGBM
lightgbm_model = lgb.LGBMRegressor(random_state=42, force_col_wise=True)

plt.subplot(2, 2, 3)
plot_validation_curves(lightgbm_model, X_train, y_train, "n_estimators", n_estimators_range,
                       "LightGBM: Криві валідації для n_estimators")

plt.subplot(2, 2, 4)
plot_validation_curves(lightgbm_model, X_train, y_train, "learning_rate", learning_rate_range,
                       "LightGBM: Криві валідації для learning_rate", xscale='log')

plt.tight_layout()
plt.show()

# Аналіз важливості ознак для XGBoost
xgboost_model.set_params(n_estimators=100, learning_rate=0.1)
xgboost_model.fit(X_train, y_train)
xgb_importance = xgboost_model.feature_importances_

plt.figure(figsize=(10, 6))
plt.bar(X_train.columns, xgb_importance)
plt.title("Важливість ознак (XGBoost)")
plt.ylabel("Оцінка важливості")
plt.xlabel("Ознаки")
plt.xticks(rotation=45)
plt.grid()
plt.show()

# Аналіз важливості ознак для LightGBM
lightgbm_model.set_params(n_estimators=100, learning_rate=0.1)
lightgbm_model.fit(X_train, y_train)
lgb_importance = lightgbm_model.feature_importances_

plt.figure(figsize=(10, 6))
plt.bar(X_train.columns, lgb_importance)
plt.title("Важливість ознак (LightGBM)")
plt.ylabel("Оцінка важливості")
plt.xlabel("Ознаки")
plt.xticks(rotation=45)
plt.grid()
plt.show()

# Оцінка XGBoost
xgboost_model.set_params(n_estimators=100, learning_rate=0.1)
xgboost_model.fit(X_train, y_train)

y_train_pred_xgb = xgboost_model.predict(X_train)
y_test_pred_xgb = xgboost_model.predict(X_test)

mse_test_xgb = mean_squared_error(y_test, y_test_pred_xgb)
print(f"XGBoost - MSE: {mse_test_xgb:.4f}")

# Оцінка LightGBM
lightgbm_model.set_params(n_estimators=100, learning_rate=0.1)
lightgbm_model.fit(X_train, y_train)

y_train_pred_lgb = lightgbm_model.predict(X_train)
y_test_pred_lgb = lightgbm_model.predict(X_test)

mse_test_lgb = mean_squared_error(y_test, y_test_pred_lgb)
print(f"LightGBM - MSE: {mse_test_lgb:.4f}")
