from tqdm import tqdm
import numpy as np
import sklearn as sk
import pandas as pd
from sklearn.datasets import load_breast_cancer
import sklearn.model_selection
import matplotlib.pyplot as plt
import random
import seaborn as sns

from ucimlrepo import fetch_ucirepo


# utils
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


def draw_loss(L_vals):
    plt.plot(list(range(len(L_vals))), L_vals, '-o', color='blue')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('Iterations VS Loss')
    plt.show()


def softmax(u):
    # return np.exp(u) / np.sum(np.exp(u))
    # use log_softmax instead to prevent overflow
    return np.exp(u - np.max(u) - np.log(np.sum(np.exp(u - np.max(u)))))


def cross_entropy(p, q, eps=1e-10):
    q = np.clip(q, eps, 1 - eps)
    return -p @ np.log(q)


def Xhat(X):
    if len(X.shape) == 1:
        return np.insert(X, 0, 1)
    return np.column_stack((np.ones((X.shape[0], 1)), X))


def grad_L_m(X, y, beta):
    if len(X.shape) == 1:
        return np.outer(Xhat(X), (softmax(Xhat(X) @ beta) - y))
    return (np.transpose(Xhat(X)) @ (softmax(Xhat(X) @ beta) - y)) / X.shape[0]


def eval_L_m(X, y, beta):
    return np.average([cross_entropy(y[index], softmax(xi @ beta)) for index, xi in enumerate(Xhat(X))])


def train_model_using_grad_descent_multi(X, y, alpha, max_iter):
    beta = np.zeros((X.shape[1] + 1, y.shape[1])).astype("float64")
    L_vals = []
    for _ in tqdm(range(max_iter)):
        beta = beta - alpha * grad_L_m(X, y, beta)
        L_vals.append(eval_L_m(X, y, beta))
    return beta, L_vals


def MSE(Yhat, Y):
    n = Yhat.shape[0]
    res = (Yhat - Y) ** 2

    return sum(res) / n


def L2gradient(X, Y, Yhat):
    # Also applicable for Log gradient
    return (X.T @ (Yhat - Y)) / X.shape[0]


def onehot(arr):
    unique_classes, inverse = np.unique(arr, return_inverse=True)
    arr = np.zeros((len(arr), len(unique_classes)))
    arr[np.arange(len(arr)), inverse] = 1

    return arr


def zerohot(df, axis=0):
    # input: array or dataframe that is one-hot encoded or simply categorical, the axis along which we decode (default = 0)
    # returns: the argmax value of the one-hot encoded columns/axes
    unhot = np.argmax(df, axis)
    return unhot


def cMatrix_log(Yhat, Y, axis=(0, 1), onehot=False):
    # logistic regression confusion matrix
    # Inputs Yhat: predicted values, Y: expected values. Yhat and Y can be either lists, np.arrays.
    # Axis tuple to precise along which axis of the matrix we decode for Yhat, Y respectively
    # returns df of confusion matrix
    if not onehot:
        d = {'Predicted Values': Yhat, 'Actual Values': Y}
        df = pd.DataFrame(data=d)

        cMat = pd.crosstab(df["Predicted Values"], df['Actual Values'])
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        ax.set_title('Confusion Matrix for Logistic Regression')
        sns.heatmap(cMat, annot=True)

    else:
        d = {'Predicted Values': zerohot(Yhat, axis[0]), 'Actual Values': zerohot(Y, axis[1])}
        df = pd.DataFrame(data=d)

        cMat = pd.crosstab(df["Predicted Values"], df['Actual Values'])
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        ax.set_title('Confusion Matrix for Logistic Regression')
        sns.heatmap(cMat, annot=True)

    # The code that follows will only run after all plots have been drawn
    plt.show()
    return cMat


def printMetrics(cMat):
    metricsList = metrics_cal(cMat)
    print("Metrics for avg_accuracy: %f, avg_precision: %f, avg_recall: %f, avg_F1: %f" % (
        metricsList[0], metricsList[1], metricsList[2], metricsList[3]))


def metrics_cal(cMat):
    # Initialize the avg of metrics
    accuracy = precision = recall = F1 = 0
    for idx in range(cMat.shape[0]):
        TP = np.sum(cMat.iloc[idx, idx])
        FP = np.sum(cMat.iloc[idx, :]) - TP
        FN = np.sum(cMat.iloc[:, idx]) - TP
        TN = np.sum(cMat.sum()) - TP - FP - FN

        accuracy += (TP + TN) / (TP + TN + FP + FN)
        precision += TP / (TP + FP)
        recall += TP / (TP + FN)
        F1 += 2 * precision * recall / (precision + recall)
    return [num/cMat.shape[0] for num in [accuracy, precision, recall, F1]]



def metricsWrapper(y_train_hat, y_train, y_test_hat, y_test):
    print("Overall Accuracy for training set: ", str(np.average(y_train_hat == y_train)))
    print("Performance metrics for training set: ")
    printMetrics(cMatrix_log(y_train_hat, y_train))

    print("==========================")
    print("Overall Accuracy for test set: ", str(np.average(y_test_hat == y_test)))
    print("Performance metrics for test set:")
    printMetrics(cMatrix_log(y_test_hat, y_test))


def split_indices(indices, N=5):
    np.random.shuffle(indices)
    return np.array_split(indices, N)


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
# draw_loss(logReg.L_vals)
# performance metrics
y_train_hat = model.predict(X_train)
y_test_hat = model.predict(X_test)

metricsWrapper(y_train_hat, y_train['class'].values, y_test_hat, y_test['class'].values)

metrics = [[], []]
folds_indices = split_indices(wine_df.index.tolist())
for idx, indices in enumerate(split_indices(wine_df.index.tolist())):
    X_train = X.iloc[np.concatenate(folds_indices[:idx] + folds_indices[idx + 1:])]
    X_test = X.iloc[indices.tolist()]
    y_train = y.iloc[np.concatenate(folds_indices[:idx] + folds_indices[idx + 1:])]
    y_test = y.iloc[indices.tolist()]

    model = LogisticRegression(alpha=0.01, max_iter=70)
    logReg = model.fit(X_train, y_train)
    # draw_loss(logReg.L_vals)
    y_train_hat = model.predict(X_train)
    y_test_hat = model.predict(X_test)
    #metrics[0].append(metrics_cal(cMatrix_log(y_train_hat, y_train)))
    #metrics[1].append(metrics_cal(cMatrix_log(y_test_hat, y_test)))
