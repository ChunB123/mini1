import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import time


def preprocessing(path):
    # Load the csv file from the path given
    df = pd.read_csv(path)

    # Check for missing or malformed values (represented as '?')
    # Remove rows with missing or malformed features
    df = df[df.ne('?').all(axis=1)]

    # Handle one-hot encoding for CHAS and RAD features
    df = pd.get_dummies(df, columns=['CHAS', 'RAD'], drop_first=True)

    # Remove the "B" column from the DataFrame
    df = df.drop("B", axis=1)

    return df


def min_max_normalize_column(column):
    min_val = column.min()
    max_val = column.max()
    if max_val != min_val:
        return (column - min_val) / (max_val - min_val)
    else:
        return column


def z_score_normalize_column(column):
    mean = column.mean()
    std_dev = column.std()
    if std_dev != 0:
        return (column - mean) / std_dev
    else:
        return column


class LinearRegressionModel:
    def __init__(self, add_bias=True, alpha=0.0, use_regularization=False):
        self.coefficients = None
        self.add_bias = add_bias
        self.alpha = alpha
        self.use_regularization = use_regularization

    def fit(self, X, y):
        # Add a column of ones for the intercept term (for bias usage x1 = 1)
        if self.add_bias:
            X = np.column_stack((np.ones(X.shape[0]), X))

        if not self.use_regularization:
            # Calculate coefficients using the analytical solution (w* = (X^T. X)^-1. X^T. y)
            self.coefficients = np.linalg.inv(X.T @ X) @ X.T @ y
        else:
            # With Ridge regularization
            n_features = X.shape[1]
            identity_matrix = np.identity(n_features)
            A = X.T @ X + self.alpha * identity_matrix
            b = X.T @ y
            self.coefficients = np.linalg.solve(A, b)

    def predict(self, X):

        # Add a column of ones for the intercept term (for bias usage x1 = 1)
        if self.add_bias:
            X = np.column_stack((np.ones(X.shape[0]), X))

        # Make predictions
        y_pred = X @ self.coefficients
        return y_pred

    def cal_mean_square_error(self, y_test, y_pred):
        return np.mean((y_test - y_pred) ** 2)

    def cal_mean_absolute_error(self, y_test, y_pred):
        return np.mean(np.abs(y_test - y_pred))

    def cal_r_squared(self, y_test, y_pred):
        # Calculate the total sum of squares
        total_sum_of_squares = np.sum((y_test - np.mean(y_test)) ** 2)

        # Calculate the residual sum of squares
        residual_sum_of_squares = np.sum((y_test - y_pred) ** 2)

        # Calculate R-squared
        r_squared = 1 - (residual_sum_of_squares / total_sum_of_squares)

        return r_squared


def get_mse_from_linear_regression(X, y, train_size=0.8, random_state=42):
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, random_state=random_state)

    # Initialize and fit the linear regression model
    model = LinearRegressionModel()
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Calculate the mean squared error (MSE)
    mse = model.cal_mean_square_error(y_pred=y_pred, y_test=y_test)

    return mse


def select_random_rows(arr, n):
    indices = np.random.choice(arr.shape[0], n, replace=False)
    return arr[indices]


def gs_func(x, u, s=1):
    return np.exp(-np.sum((x - u) ** 2) / (2 * (s ** 2)))


# Take numpy arr as input
class GSSampller:
    def __init__(self, n_components):
        self.random_rows = None
        self.n_components = n_components

    def transformed_arr_builder(self, original_arr):
        # Example: Input: N X 19, Output N X 5
        transformed_arr = np.zeros((original_arr.shape[0], self.n_components))
        for col_idx, random_row in enumerate(self.random_rows):
            for row_idx, x_row in enumerate(original_arr):
                transformed_arr[row_idx][col_idx] = gs_func(x_row, random_row)
        return transformed_arr

    def fit_transform(self, X_train):
        self.random_rows = select_random_rows(X_train, self.n_components)
        return self.transformed_arr_builder(X_train)

    def transform(self, X_test):
        return self.transformed_arr_builder(X_test)


# Define the file path
dataset_path = "/mini1/boston.csv"
housing_df = preprocessing(dataset_path)
X = z_score_normalize_column(housing_df.drop(columns=['MEDV']).values.astype('float64'))
y = housing_df['MEDV'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)

model = LinearRegressionModel()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = model.cal_mean_square_error(y_pred=y_pred, y_test=y_test)

# Conduct feature extraction using 5 Gaussian functions
gsMapper = GSSampller(n_components=5)
X_train_new = gsMapper.fit_transform(X_train)
X_test_new = gsMapper.transform(X_test)

model = LinearRegressionModel()
model.fit(X_train_new, y_train)
y_pred = model.predict(X_test_new)
mse_transformed = model.cal_mean_square_error(y_pred=y_pred, y_test=y_test)


# Experiment with Gaussian basis functions
def plot_metrics_with_n_gs_funcitons(gs_dic, metricsList: list):
    plt.figure(figsize=(10, 6))
    for metric in metricsList:
        plt.plot(gs_dic['n_components'], gs_dic[metric], label=metric, marker='o')

    plt.xlabel('n_components')
    plt.ylabel('mse')
    plt.legend()
    plt.grid(True)
    plt.show()


def gs_grid_search(X_train, y_train, num_funcs: list):
    gs_dic = {
        'n_components': [],
        'mse': []
    }
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, train_size=0.8)
    for n in num_funcs:
        gsMapper = GSSampller(n_components=n)
        X_train_new = gsMapper.fit_transform(X_train)
        X_test_new = gsMapper.transform(X_test)

        model = LinearRegressionModel()
        model.fit(X_train_new, y_train)
        y_pred = model.predict(X_test_new)
        gs_dic['n_components'].append(n)
        gs_dic['mse'].append(model.cal_mean_square_error(y_pred=y_pred, y_test=y_test))
    plot_metrics_with_n_gs_funcitons(gs_dic, ['mse'])
    return pd.DataFrame(gs_dic)


gs_grid_search(X_train, y_train, [5, 10, 20, 40, 60, 65, 70, 75, 80, 100, 120])

# Apply the best nums (65) of GS functions to do feature extraction and test performance in test set
gsMapper = GSSampller(n_components=65)
X_train_new = gsMapper.fit_transform(X_train)
X_test_new = gsMapper.transform(X_test)

model = LinearRegressionModel()
model.fit(X_train_new, y_train)
y_pred = model.predict(X_test_new)
mse_transformed_final = model.cal_mean_square_error(y_pred=y_pred, y_test=y_test)


