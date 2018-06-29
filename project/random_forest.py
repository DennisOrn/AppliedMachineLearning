import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the results dataset into a dataframe.
df = pd.read_csv('data/results.csv')

# Extract the results from the 2017 season.
df_2017 = df[(df.raceId >= 969) & (df.raceId <= 988)]

# Train with the first 19 races, test with the last race.
train = df_2017[:-20]
test = df_2017[-20:]

# Try to predict the position.
X_train = train.drop(columns=['position'])
X_train = X_train.drop(columns=['positionText', 'time', 'fastestLapTime', 'fastestLapSpeed'])
y_train = train.position

X_test = test.drop(columns=['position'])
X_test = X_test.drop(columns=['positionText', 'time', 'fastestLapTime', 'fastestLapSpeed'])
y_test = test.position

# Replace all NaN's with 0.
X_train = X_train.fillna(0)
y_train = y_train.fillna(0)

X_test = X_test.fillna(0)
y_test = y_test.fillna(0)

# Create the classifier.
clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train, y_train)

# Make a prediction and print the accuracy.
prediction = clf.predict(X_test)
prediction = [round(value) for value in prediction]
print(f'prediction: {prediction}')
print(f'reality:    {y_test.tolist()}')

accuracy = accuracy_score(y_test, prediction)
print(f'accuracy:   {accuracy * 100}%')
