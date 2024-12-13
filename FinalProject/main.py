import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold

from preprocessing import load_and_preprocess_data
from utils import plot_validation_curve
from modeling import evaluate_model
from models import (
    train_decision_tree, hyperparams_decision_tree,
    train_xgb, hyperparams_xgb,
    train_knn, hyperparams_knn
)

MODELS = [
    {
        "name": "DecisionTreeClassifier",
        "train_func": train_decision_tree,
        "hyperparams_func": hyperparams_decision_tree
    },
    {
        "name": "XGBoostClassifier",
        "train_func": train_xgb,
        "hyperparams_func": hyperparams_xgb
    },
    {
        "name": "KNeighborsClassifier",
        "train_func": train_knn,
        "hyperparams_func": hyperparams_knn
    }
]


def process_models(model_info, X_train, y_train, X_test, y_test, kf):
    name = model_info["name"]
    train_func = model_info["train_func"]
    hyperparams_func = model_info["hyperparams_func"]
    # Навчання базової моделі
    model = train_func(X_train, y_train)
    print(f"Initial {name}:")
    evaluate_model(model, X_test, y_test, name)
    # Тюнінг моделі
    grid, params = hyperparams_func(model, X_train, y_train, kf)
    print(f"Using hyperparams {name}:")
    evaluate_model(grid.best_estimator_, X_test, y_test, f"{name} (with hyperparams)")
    # Побудова валідаційних кривих
    for param_name, param_values in params.items():
        plot_validation_curve(grid, param_name, param_values)


def main():
    # Завантаження і препроцесинг даних
    X, y, df = load_and_preprocess_data('../data/vodafone_music_subset.csv')
    # Аналіз даних
    sns.countplot(x='target', data=df)
    plt.title('Розподіл target')
    plt.show()
    # Розбиття вибірки
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    kf = KFold(n_splits=4, shuffle=True, random_state=42)
    # Обробка всіх моделей у циклі
    for model_info in MODELS:
        process_models(model_info, X_train, y_train, X_test, y_test, kf)


if __name__ == '__main__':
    main()
