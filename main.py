from tqdm import tqdm
import numpy as np
import sklearn as sk
import pandas as pd
from sklearn.datasets import load_breast_cancer
import sklearn.model_selection
import matplotlib.pyplot as plt
import random

from ucimlrepo import fetch_ucirepo


def train_model_using_grad_descent(X, y, alpha=0.1, max_iter=100):
    beta = np.zeros(X.shape[1] + 1)
    L_vals = []
    for _ in tqdm(range(max_iter)):
        beta = beta - alpha * grad_L(X, y, beta)
        L_vals.append(eval_L(X, y, beta))
    return beta, L_vals


def train_model_using_stochastic_grad_descent(X, y, alpha=0.1, epoch=500):
    beta = np.zeros(X.shape[1] + 1)
    L_vals = []
    for _ in tqdm(range(epoch)):
        indexList = random.sample(range(X.shape[0]), X.shape[0])
        for i in indexList:
            beta = beta - alpha * grad_L(X[i], y[i], beta)
        L_vals.append(eval_L(X, y, beta))
    return beta, L_vals


def draw(datas, legends, xlabel, ylabel, title):
    # create a new figure
    fig, ax = plt.subplots()
    # Plot each curve on the axes object
    for index, data in enumerate(datas):
        ax.plot(data, label=str(legends[index]))

    # Add a legend to the axes object
    ax.legend()

    # Set the x-axis and y-axis labels
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # Set the title of the figure
    ax.set_title(title)

    # Display the figure
    plt.show()


def draw_loss(L_vals):
    plt.plot(list(range(len(L_vals))), L_vals, '-o', color='blue')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('Iterations VS Loss')
    plt.show()


def normalize_features(df, method='z-score'):
    # Extract the feature columns (excluding 'MEDV')
    feature_cols = [col for col in df.columns if col != 'MEDV']

    if method == 'z-score':
        # Z-score normalization
        means = df[feature_cols].mean()
        std_devs = df[feature_cols].std()

        # Apply z-score normalization to the feature columns
        normalized_df = df.copy()
        normalized_df[feature_cols] = (df[feature_cols] - means) / std_devs
    elif method == 'min-max':
        # Min-max scaling
        min_val = df[feature_cols].min()
        max_val = df[feature_cols].max()

        # Apply Min-max scaling normalization to the feature columns
        normalized_df = df.copy()
        normalized_df[feature_cols] = (df[feature_cols] - min_val) / (max_val - min_val)
    else:
        raise ValueError("Invalid normalization method.")

    return normalized_df


def sigmoid(u):
    # use identity to prevent overflow
    if isinstance(u, (int, float)):
        if u >= 0:
            return 1.0 / (1.0 + np.exp(-u))
        else:
            return np.exp(u) / (1.0 + np.exp(u))
    else:
        positive_mask = u >= 0
        output = np.array([0 for x in range(len(u))]).astype("float64")
        output[positive_mask] = 1 / (1 + np.exp(-u[positive_mask]))
        output[~positive_mask] = np.exp(u[~positive_mask]) / (1 + np.exp(u[~positive_mask]))
        return output


def softmax(u):
    # return np.exp(u) / np.sum(np.exp(u))
    # use log_softmax instead to prevent overflow
    return np.exp(u - np.max(u) - np.log(np.sum(np.exp(u - np.max(u)))))


def binary_cross_entropy(p, q, eps=1e-10):
    # prevent ln(0)
    q = np.clip(q, eps, 1 - eps)
    return -p * np.log(q) - (1 - p) * np.log(1 - q)


def cross_entropy(p, q, eps=1e-10):
    q = np.clip(q, eps, 1 - eps)
    return -p @ np.log(q)


def Xhat(X):
    if len(X.shape) == 1:
        return np.insert(X, 0, 1)
    return np.column_stack((np.ones((X.shape[0], 1)),X))


def grad_L(X, y, beta):
    if len(X.shape) == 1:
        return (sigmoid(Xhat(X) @ beta) - y) * Xhat(X)
    return np.average(np.array([(sigmoid(x @ beta) - y[index]) * x for index, x in enumerate(Xhat(X))]), axis=0)


def eval_L(X, y, beta):
    return np.average([binary_cross_entropy(y[index], sigmoid(xi @ beta)) for index, xi in enumerate(Xhat(X))])


def grad_L_m(X, y, beta):
    if len(X.shape) == 1:
        return np.outer(Xhat(X), (softmax(Xhat(X) @ beta) - y))
    return (np.transpose(Xhat(X)) @ (softmax(Xhat(X) @ beta) - y)) / X.shape[0]


def eval_L_m(X, y, beta):
    return np.average([cross_entropy(y[index], softmax(xi @ beta)) for index, xi in enumerate(Xhat(X))])

def train_model_using_stochastic_grad_descent_multi(X, y, alpha=0.01, epoch=500):
    beta = np.zeros((X.shape[1] + 1, y.shape[1])).astype("float64")
    L_vals = []
    for _ in tqdm(range(epoch)):
        indexList = random.sample(range(X.shape[0]), X.shape[0])
        for i in indexList:
            beta = beta - alpha * grad_L_m(X[i], y[i], beta)
        # use cross-entropy
        L_vals.append(eval_L_m(X, y, beta))
    return beta, L_vals


def train_model_using_grad_descent_multi(X, y, alpha, max_iter):
    beta = np.zeros((X.shape[1] + 1, y.shape[1])).astype("float64")
    L_vals = []
    for _ in tqdm(range(max_iter)):
        beta = beta - alpha * grad_L_m(X, y, beta)
        L_vals.append(eval_L_m(X, y, beta))
    return beta, L_vals


# Multiclass Gradient descent
class LogisticRegression:

    def __init__(self, alpha=0.01, max_iter=500):
        self.beta = None
        self.L_vals = None
        self.alpha = alpha
        self.max_iter = max_iter

    def fit(self, X_train, y_train):
        # one-hot-encoding y_train
        unique_classes, inverse = np.unique(y_train, return_inverse=True)
        y_train_hot = np.zeros((len(y_train), len(unique_classes)))
        y_train_hot[np.arange(len(y_train)), inverse] = 1

        self.beta, self.L_vals = train_model_using_grad_descent_multi(X_train, y_train_hot, self.alpha, self.max_iter)
        return self

    def predict(self, X_test):
        # first class in y_pred is class 2 in Y
        y_pred = [np.argmax(yi) + 1 for yi in (Xhat(X_test) @ self.beta)]
        return y_pred

# Split Dataframe to N folds
def split_dataframe(df, n_folds=5, random_seed=10):
    df_shuffled = df.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    fold_size = len(df_shuffled) // n_folds

    folds = []
    for i in range(n_folds):
        if i == n_folds - 1:
            # Store remaining data to last fold
            folds.append(df_shuffled.iloc[i*fold_size:])
        else:
            folds.append(df_shuffled.iloc[i*fold_size:(i+1)*fold_size])

    return folds

# Fetch dataset
wine = fetch_ucirepo(id=109)

# Create DataFrames for features (X) and classification class (Y)
X = pd.DataFrame(data=wine.data.features)
y = pd.DataFrame(data=wine.data.targets)
wine_df = pd.concat([X, y], axis=1)

# Check for missing or malformed values (represented as '?')
# Remove rows with missing or malformed features
wine_df = wine_df[wine_df.ne('?').all(axis=1)]

X = normalize_features(X)
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(alpha=0.01, max_iter=70)
logReg = model.fit(X_train, y_train)
#draw_loss(logReg.L_vals)
y_pred = model.predict(X_test)
print("Accuracy using Gradient Descent: ", str(np.average(y_pred == y_test['class'].values)))

folds = split_dataframe(wine_df)

for idx, fold in enumerate(folds):
    data_train = pd.concat(folds[:idx] + folds[idx+1:])
    data_val = fold
    model = LogisticRegression(alpha=0.01, max_iter=70)
    logReg = model.fit(X_train, y_train)