from sklearn.datasets import load_iris, load_wine
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

dataset = load_iris()
X = dataset["data"]
y = dataset["target"]

def nearest_neighbor_classifier(X, y, x, k):
    distances = [calc_distance(x, X[i]) for i in range(len(X))]
    pairs = list(zip(distances, y))
    pairs = sorted(pairs, key=lambda x: x[0])[0:k]
    nearest_y = [pair[1] for pair in pairs]

    y_counts = {}
    for v in nearest_y:
        y_counts[v] = y_counts.get(v, 0) + 1

    res = sorted(y_counts.items(), key=lambda x: x[1],reverse = True)
    return res[0][0],{i[0]:(i[1]/k) for i in res}

def calc_distance(u, v):
    return np.sqrt(np.sum((u - v)**2))


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

hits = 0
for i in range(len(X_test)):
    x = X_test[i]
    y_true = y_test[i]
    y_pred,fromalg = nearest_neighbor_classifier(X_train, y_train, x, 9)  #i don't know how to use probabilities
    print(fromalg)
    if y_true == y_pred:
        hits += 1

print("{} out of {} are correct".format(hits, len(X_test)))
