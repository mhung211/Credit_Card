import numpy as np
import random

def train_val_test_split(X, y, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, shuffle=True):
    m = X.shape[0]

    if shuffle:
        idx = np.random.permutation(m)
        X = X[idx]
        y = y[idx]

    train_end = int(train_ratio * m)
    val_end = train_end + int(val_ratio * m)

    X_train = X[:train_end]
    y_train = y[:train_end]

    X_val = X[train_end:val_end]
    y_val = y[train_end:val_end]

    X_test = X[val_end:]
    y_test = y[val_end:]

    return X_train, y_train, X_val, y_val, X_test, y_test

def sigmoid(z):
    return 1/(1 + np.exp(-z))

def predict(X, theta):
    z = np.dot(X, theta)
    return sigmoid(z)

def compute_loss(y_hat, y):
    return -y.T.dot(np.log(y_hat)) - (1-y).T.dot(np.log(1 - y_hat))

def compute_gradient(X, y_hat, y):
    return X.T.dot(y_hat - y)

def update_gradient(theta, learning_rate, gradient):
    theta -= learning_rate*gradient
    return theta

def predict_label(y_hat, threshold = 0.3):
    y_pred = (y_hat >= threshold).astype(int)
    return y_pred

def accuracy(y_hat, y):
    y_pred = predict_label(y_hat)
    return np.mean(y == y_pred)

def evaluate(y_pred, y):
    TP = np.sum((y == 1) & (y_pred == 1))
    TN = np.sum((y == 0) & (y_pred == 0))
    FP = np.sum((y == 0) & (y_pred == 1))
    FN = np.sum((y == 1) & (y_pred == 0))

    precision = TP / (TP + FP) if (TP + FP) != 0 else 0
    recall = TP / (TP + FN) if (TP + FN) != 0 else 0

    if precision + recall == 0:
        f1_score = 0
    else:
        f1_score = 1 / (1/precision + 1/recall)

    confu_matrix = np.array([
        [TN, FP],
        [FN, TP]
    ])
    
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1-score: {f1_score}")
    print(f"Confusion matrix: {confu_matrix}")

def minmax_scaler(X):
    X = X.astype(float)

    X_min = X.min(axis=0)
    X_max = X.max(axis=0)

    diff = X_max - X_min
    diff[diff == 0] = 1  

    X_scaled = (X - X_min) / diff
    return X_scaled