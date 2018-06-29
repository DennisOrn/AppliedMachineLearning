import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


features = pd.read_csv('data/results.csv')
# features = pd.get_dummies(features)
# features = features.head(5)
print(features)


features = features.drop('fastestLapTime', axis = 1)
features = features.drop('time', axis = 1)
labels = np.array(features['position'])
# feature_list = list(features.columns)
# features = np.array(features)
print(features)



train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.25, random_state = 42)

clf = RandomForestClassifier(n_jobs=2, random_state=0)
clf.fit(train_features, train_labels)

# predictions = clf.predict(test_features)
# print(predictions)
# errors = abs(predictions - test_labels)
# print(errors)