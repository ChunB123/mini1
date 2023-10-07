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


def grad_L_m_L1(X, y, beta, lambda_val=0):
    # Gradient with L1 regularization term
    original_gradient = (np.transpose(Xhat(X)) @ (softmax(Xhat(X) @ beta) - y)) / X.shape[0]
    # No penalization for intercept
    original_gradient[1:] += lambda_val * np.sign(beta[1:])
    return original_gradient


def eval_L_m_L1(X, y, beta, lambda_val=0):
    # Including L1 regularization term
    l1_penalty = lambda_val * np.sum(np.abs(beta))
    return np.average([cross_entropy(y[index], softmax(xi @ beta)) for index, xi in enumerate(Xhat(X))]) + l1_penalty


def train_model_using_grad_descent_multi(X, y, alpha, max_iter, eps):
    beta = np.zeros((X.shape[1] + 1, y.shape[1])).astype("float64")
    L_vals = []
    for _ in tqdm(range(max_iter)):
        grad_matrix = grad_L_m(X, y, beta)
        if np.linalg.norm(grad_matrix) < eps:
            break
        beta = beta - alpha * grad_matrix
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
    accuracy, precision, recall, F1 = metrics_cal(cMat)
    print("Metrics for avg_accuracy: %f, avg_precision: %f, avg_recall: %f, avg_F1: %f" % (
        accuracy, precision, recall, F1))


def cMat_builder(Yhat, Y):
    df = pd.DataFrame(data={'Predicted Values': Yhat, 'Actual Values': Y})
    return pd.crosstab(df["Predicted Values"], df['Actual Values'])


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
        if precision + recall != 0:
            F1_temp = 2 * precision * recall / (precision + recall)
        else:
            F1_temp = 0.0

        F1 += F1_temp
    return accuracy / cMat.shape[0], precision / cMat.shape[0], recall / cMat.shape[0], F1 / cMat.shape[0]


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


def plot_metrics_with_diff_training_size(metrics_df_ts, metricsList: list):
    plt.figure(figsize=(10, 6))
    for metric in metricsList:
        plt.plot(metrics_df_ts['training_size'], metrics_df_ts[metric], label=metric, marker='o')

    plt.xlabel('Training Size')
    plt.ylabel('Metric Value')
    plt.title('Learning Curves')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_metrics_with_diff_learning_rates(metrics_df_lr, metricsList: list):
    plt.figure(figsize=(10, 6))
    for metric in metricsList:
        plt.plot(metrics_df_lr['learning_rates'], metrics_df_lr[metric], label=metric, marker='o')

    plt.xlabel('learning_rates')
    plt.ylabel('Metric Value')
    plt.title('Learning Curves')
    plt.legend()
    plt.grid(True)
    plt.show()


# Multiclass Gradient descent
class LogisticRegression:

    def __init__(self, alpha=0.01, max_iter=500, eps=1e-2):
        self.beta = None
        self.L_vals = None
        self.alpha = alpha
        self.max_iter = max_iter
        self.eps = eps

    def fit(self, X_train, y_train):
        # one-hot-encoding y_train
        unique_classes, inverse = np.unique(y_train, return_inverse=True)
        y_train_hot = np.zeros((len(y_train), len(unique_classes)))
        y_train_hot[np.arange(len(y_train)), inverse] = 1

        self.beta, self.L_vals = train_model_using_grad_descent_multi(X_train, y_train_hot, self.alpha, self.max_iter,
                                                                      self.eps)
        return self

    def predict(self, X_test):
        # first class in y_pred is class 2 in Y
        y_pred = [np.argmax(yi) + 1 for yi in (Xhat(X_test) @ self.beta)]
        return y_pred

def train_model_using_grad_descent_multi_L1(X, y, alpha, max_iter, eps, lambda_val=0):
    beta = np.zeros((X.shape[1] + 1, y.shape[1])).astype("float64")
    L_vals = []
    for _ in tqdm(range(max_iter)):
        grad_matrix = grad_L_m_L1(X, y, beta, lambda_val)
        if np.linalg.norm(grad_matrix) < eps:
            break
        beta = beta - alpha * grad_matrix
        L_vals.append(eval_L_m_L1(X, y, beta, lambda_val))
    return beta, L_vals
class LogisticRegression_L1:
    def __init__(self, alpha=0.01, max_iter=500, eps=1e-2, lambda_val=0.1):
        self.beta = None
        self.L_vals = None
        self.alpha = alpha
        self.max_iter = max_iter
        self.eps = eps
        self.lambda_val = lambda_val  # L1 regularization strength

    def fit(self, X_train, y_train):
        # one-hot-encoding y_train
        unique_classes, inverse = np.unique(y_train, return_inverse=True)
        y_train_hot = np.zeros((len(y_train), len(unique_classes)))
        y_train_hot[np.arange(len(y_train)), inverse] = 1

        self.beta, self.L_vals = train_model_using_grad_descent_multi_L1(X_train, y_train_hot, self.alpha, self.max_iter,
                                                                      self.eps, self.lambda_val)
        return self

    def predict(self, X_test):
        # first class in y_pred is class 2 in Y
        y_pred = [np.argmax(yi) + 1 for yi in (Xhat(X_test) @ self.beta)]
        return y_pred


# Multiclass Gradient Descent with SGD
class logReg_SGD:
    def __init__(self, alpha=0.01, max_iter=500, eps=0.001, size=None):
        # Size parameter is asking for the size of the weight matrix
        # Which is usually D x C (features x categories)
        # The Y expectation matrix should be one-hot encoded before being inputted
        self.w = np.random.randn(size[0], size[1])
        self.max_iter = max_iter
        self.alpha = alpha
        self.cost = []
        self.eps = eps
        self.grad = 1

    # Feed the X data in a forward loop
    def forwardpass(self, X):
        a = np.dot(X, self.w)
        return a

    def SGD(self, X, Y, epochs, mini_batch_size, test_data=None):
        m, n = X.shape
        if test_data:
            Xtest, Ytest = test_data
            Xtest = Xtest.values

        for i in tqdm(range(1, epochs + 1)):
            if np.linalg.norm(self.grad) < self.eps: break;
            temp = list(zip(X.values, Y))
            random.shuffle(temp)
            Xshuffled, Yshuffled = zip(*temp)
            Xshuffled, Yshuffled = np.array(Xshuffled), np.array(Yshuffled)

            for k in range(0, m, mini_batch_size):
                Xmini, Ymini = Xshuffled[k:k + mini_batch_size], Yshuffled[k:k + mini_batch_size]
                self.update_mini_batch(Xmini, Ymini, self.alpha)
            # if test_data:
            #         print("Epoch {0}: {1} / {2}".format(
            #             i, self.eval(test_data = test_data), len(Xtest)))
            # else:
            #         print("Epoch {0} complete".format(i))

    def update_mini_batch(self, X, Y, alpha):
        yhat = self.forwardpass(X)
        grad = L2gradient(X, Y, yhat)
        self.w = self.w - alpha * grad
        self.grad = np.mean(grad)

    def eval(self, test_data=None):
        Xtest, Ytest = test_data
        prediction = self.forwardpass(Xtest)
        predictions = np.argmax(prediction, axis=1)
        expected = np.argmax(Ytest, axis=1)
        return sum(predictions == expected)

    def predict(self, Xtest):
        prediction = self.forwardpass(Xtest)
        out = softmax(prediction)

        return np.argmax(prediction, axis=1)


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

model = LogisticRegression(alpha=0.1, max_iter=70)
logReg = model.fit(X_train, y_train)
# draw_loss(logReg.L_vals)
# performance metrics
y_train_hat = model.predict(X_train)
y_test_hat = model.predict(X_test)

metricsWrapper(y_train_hat, y_train['class'].values, y_test_hat, y_test['class'].values)

# 3.3.2
# train_accuracy, train_precision, train_recall, train_F1, test_accuracy, test_precision, test_recall, test_F1
metrics = [0 for x in range(8)]
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
    train_accuracy, train_precision, train_recall, train_F1 = metrics_cal(
        cMat_builder(y_train_hat, y_train['class'].values))
    test_accuracy, test_precision, test_recall, test_F1 = metrics_cal(cMat_builder(y_test_hat, y_test['class'].values))
    metrics[0] += train_accuracy
    metrics[1] += train_precision
    metrics[2] += train_recall
    metrics[3] += train_F1
    metrics[4] += test_accuracy
    metrics[5] += test_precision
    metrics[6] += test_recall
    metrics[7] += test_F1

metrics = [metric / len(folds_indices) for metric in metrics]
print("")
print(
    "Metrics for CV: train_accuracy, train_precision, train_recall, train_F1, test_accuracy, test_precision, test_recall, test_F1")
print(metrics)

# 3.3.3
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.2, random_state=42)
train_sizes = np.linspace(0.2, 0.8, 7)

metrics_dic = {
    'training_size': [],
    'train_accuracy': [],
    'train_precision': [],
    'train_recall': [],
    'train_F1': [],
    'test_accuracy': [],
    'test_precision': [],
    'test_recall': [],
    'test_F1': []
}

for size in train_sizes:
    # resize the X_train, y_train
    X_train_temp, _, y_train_temp, _ = sklearn.model_selection.train_test_split(X_train, y_train, train_size=size,
                                                                                random_state=11)

    # Train Gradient Descent model
    model = LogisticRegression(alpha=0.01, max_iter=70)
    logReg = model.fit(X_train_temp, y_train_temp)

    y_train_hat = model.predict(X_train_temp)
    y_test_hat = model.predict(X_test)
    train_accuracy, train_precision, train_recall, train_F1 \
        = metrics_cal(cMat_builder(y_train_hat, y_train_temp['class'].values))
    test_accuracy, test_precision, test_recall, test_F1 \
        = metrics_cal(cMat_builder(y_test_hat, y_test['class'].values))

    # Compute and store metrics
    metrics_dic['training_size'].append(len(X_train_temp))
    metrics_dic['train_accuracy'].append(train_accuracy)
    metrics_dic['train_precision'].append(train_precision)
    metrics_dic['train_recall'].append(train_recall)
    metrics_dic['train_F1'].append(train_F1)
    metrics_dic['test_accuracy'].append(test_accuracy)
    metrics_dic['test_precision'].append(test_precision)
    metrics_dic['test_recall'].append(test_recall)
    metrics_dic['test_F1'].append(test_F1)

metrics_df_ts = pd.DataFrame(metrics_dic)

train_metrics = ['train_accuracy', 'train_precision', 'train_recall', 'train_F1']
test_metrics = ['test_accuracy', 'test_precision', 'test_recall', 'test_F1']

plot_metrics_with_diff_training_size(metrics_df_ts, train_metrics)
plot_metrics_with_diff_training_size(metrics_df_ts, test_metrics)

# 3.5.1
learning_rates = [0.0005, 0.005, 0.05, 0.5, 5, 50]
metrics_dic_lr = {
    'learning_rates': [],
    'train_accuracy': [],
    'train_precision': [],
    'train_recall': [],
    'train_F1': [],
    'test_accuracy': [],
    'test_precision': [],
    'test_recall': [],
    'test_F1': []
}
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.95, random_state=351)
for lr in learning_rates:
    model = LogisticRegression(alpha=lr, max_iter=50)
    logReg = model.fit(X_train, y_train)

    y_train_hat = model.predict(X_train)
    y_test_hat = model.predict(X_test)
    train_accuracy, train_precision, train_recall, train_F1 \
        = metrics_cal(cMat_builder(y_train_hat, y_train['class'].values))
    test_accuracy, test_precision, test_recall, test_F1 \
        = metrics_cal(cMat_builder(y_test_hat, y_test['class'].values))

    # Compute and store metrics
    metrics_dic_lr['learning_rates'].append(lr)
    metrics_dic_lr['train_accuracy'].append(train_accuracy)
    metrics_dic_lr['train_precision'].append(train_precision)
    metrics_dic_lr['train_recall'].append(train_recall)
    metrics_dic_lr['train_F1'].append(train_F1)
    metrics_dic_lr['test_accuracy'].append(test_accuracy)
    metrics_dic_lr['test_precision'].append(test_precision)
    metrics_dic_lr['test_recall'].append(test_recall)
    metrics_dic_lr['test_F1'].append(test_F1)

metrics_df_lr = pd.DataFrame(metrics_dic_lr)
plot_metrics_with_diff_learning_rates(metrics_df_lr, ['train_accuracy', 'train_precision', 'train_recall', 'train_F1'])
plot_metrics_with_diff_learning_rates(metrics_df_lr, ['test_accuracy', 'test_precision', 'test_recall', 'test_F1'])

# 3.6
batch_sizes = [1, 4, 8, 16, 32]
learning_rates = [0.00625, 0.0125, 0.025, 0.05, 0.075, 0.1, 0.15]
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.2, random_state=360)
configs_dic = {
    'batch_sizes': [],
    'learning_rates': [],
    'cv_F1': []
}
for batch_size in batch_sizes:
    for lr in learning_rates:
        # Use 5 folds CV to calculate the F1
        folds_indices = split_indices(wine_df.index.tolist())
        for idx, indices in enumerate(split_indices(wine_df.index.tolist())):
            X_train = X.iloc[np.concatenate(folds_indices[:idx] + folds_indices[idx + 1:])]
            X_test = X.iloc[indices.tolist()]
            y_train = y.iloc[np.concatenate(folds_indices[:idx] + folds_indices[idx + 1:])]
            y_test = y.iloc[indices.tolist()]

            ytrain, ytest = onehot(y_train), onehot(y_test)
            model_SGD = logReg_SGD(alpha=lr, size=[X_train.shape[1], ytrain.shape[1]])
            model_SGD.SGD(X_train, ytrain, epochs=1, mini_batch_size=batch_size)

            # draw_loss(logReg.L_vals)
            y_train_hat = model_SGD.predict(X_train)
            y_test_hat = model_SGD.predict(X_test)
            _, _, _, test_F1 = metrics_cal(
                cMat_builder(y_test_hat, y_test['class'].values))

            configs_dic['batch_sizes'].append(batch_size)
            configs_dic['learning_rates'].append(lr)
            configs_dic['cv_F1'].append(test_F1)

configs_dic_df = pd.DataFrame(configs_dic)

# Best combination: batch_sizes=1, lr=0.05

# Performance on Test dataset
# Testing out SGD logistic regression
ytrain, ytest = y_train, y_test
ytrain, ytest = onehot(ytrain), onehot(ytest)
model_SGD = logReg_SGD(alpha=0.05, size=[X_train.shape[1], ytrain.shape[1]])
model_SGD.SGD(X_train, ytrain, epochs=1, mini_batch_size=1, test_data=(X_test, ytest))
ypred = onehot(np.array(model.predict(X_test)))
printMetrics(cMatrix_log(ypred, ytest, axis=(1, 1), onehot=True))

# 3.9
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.2, random_state=390)
model_L1 = LogisticRegression_L1(alpha=0.01, max_iter=500, lambda_val=0.5)
logReg_L1 = model_L1.fit(X_train, y_train)
draw_loss(logReg_L1.L_vals)

avg_coef = np.mean(np.abs(logReg_L1.beta[1:]), axis=1)
coef_filtered = np.array([coef if coef > 1e-5 else 0 for coef in avg_coef])
