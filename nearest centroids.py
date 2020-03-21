from sklearn.datasets import load_iris, load_wine
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

dataset = load_wine()
X = dataset["data"]
y = dataset["target"]

def centroids(objects,coordinates):
	centers = {}
	for i in range(len(objects)):
		centers[objects[i]] = centers.get(objects[i],[])
		centers[objects[i]].append(coordinates[i])
	for i in range(len(centers)):
		centers[i] = sum(centers[i])/len(centers)
	return centers

def nearest_centroids_classifier(X, y, x):
    centers = centroids(y,X)
    distances = {i : calc_distance(x, centers[i]) for i in centers.keys()}
    distances = sorted(distances, key=distances.get)
    return distances[0]
    

def calc_distance(u, v):
    return np.sqrt(np.sum((u - v)**2))


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

hits = 0
for i in range(len(X_test)):
	x = X_test[i]
	y_true = y_test[i]
	y_pred = nearest_centroids_classifier(X_train, y_train, x)
	if y_true == y_pred:
		hits += 1

print("{} out of {} are correct".format(hits, len(X_test)))
