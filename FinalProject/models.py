import numpy as np
import xgboost as xgb
from collections import Counter
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

def train_decision_tree(X_train, y_train):
    model = DecisionTreeClassifier(random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)
    return model

def tune_decision_tree(model, X_train, y_train, kf):
    tree_params = {
        'min_samples_split': np.arange(6, 18, 3),
        'max_depth': np.arange(2, 13, 2),
        'min_samples_leaf': [1, 2, 4, 6, 8],
        'class_weight': ['balanced', None]
    }
    tree_grid = GridSearchCV(model, tree_params, cv=kf, scoring='f1', n_jobs=-1, verbose=2)
    tree_grid.fit(X_train, y_train)
    return tree_grid, tree_params

def train_xgb(X_train, y_train):
    counter = Counter(y_train)
    scale_pos_weight = counter[0] / counter[1]
    model = xgb.XGBClassifier(random_state=42, scale_pos_weight=scale_pos_weight)
    model.fit(X_train, y_train)
    return model

def tune_xgb(model, X_train, y_train, kf):
    xgb_params = {
        'n_estimators': range(100, 501, 100),  # Wider range of trees
        'max_depth': [3, 5, 7, 9],
    }
    xgb_grid = GridSearchCV(model, xgb_params, cv=kf, scoring='f1', n_jobs=-1, verbose=2)
    xgb_grid.fit(X_train, y_train)
    return xgb_grid, xgb_params

def train_knn(X_train, y_train):
    model = KNeighborsClassifier(metric='minkowski', p=2, weights='uniform')
    model.fit(X_train, y_train)
    return model

def tune_knn(model, X_train, y_train, kf):
    k_neighbors_params = {
        'n_neighbors': [3, 5, 7, 9, 11],
        'metric': ['minkowski', 'euclidean', 'manhattan'],
        'p': [1, 2],
        'weights': ['uniform', 'distance']
    }
    knn_grid = GridSearchCV(model, k_neighbors_params, cv=kf, scoring='recall', n_jobs=-1, verbose=2)
    knn_grid.fit(X_train, y_train)
    return knn_grid, k_neighbors_params