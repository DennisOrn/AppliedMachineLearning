from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier

iris = datasets.load_iris()

X = iris.data[:-1]
y = iris.target[:-1]

removedInstance = iris.data[-1:]
print(removedInstance)

classifier = KNeighborsClassifier(3)
classifier.fit(X, y)

print(classifier.predict(removedInstance))
print(classifier.predict_proba(removedInstance))