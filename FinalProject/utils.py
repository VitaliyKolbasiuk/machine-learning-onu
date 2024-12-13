import pandas as pd
import matplotlib.pyplot as plt

def plot_validation_curve(model_grid, param_name, params=None):
    results_df = pd.DataFrame(model_grid.cv_results_)
    param_column = 'param_' + param_name

    grouped = results_df.groupby(param_column)['mean_test_score'].mean()
    x_values = grouped.index
    y_values = grouped.values

    plt.plot(x_values, y_values, marker='o')
    plt.xlabel(param_name)
    plt.ylabel('Mean Test F1 Score')
    plt.title(f'Validation curve for {param_name}')
    plt.grid(True)
    plt.show()