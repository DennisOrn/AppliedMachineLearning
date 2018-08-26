import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

# Load the results dataset into a dataframe.
df = pd.read_csv('data/results.csv')

# Extract the results from 2014-2017.
df = df[(df.raceId >= 900) & (df.raceId <= 988)]

# Train with all races except the 5 last (remember: every race has 20 drivers).
train = df[:-100]

# Test with the 5 last races.
test = df[-100:]

# Try to predict the position based on a few other features.
y_train = train.positionOrder
y_test = test.positionOrder

# driverId - who the driver is.
# rank - where the driver is in the championship standings.
# grid - the starting position.
# constructorId - which team the driver is racing for.

columns = ['driverId', 'constructorId', 'grid', 'rank']
X_train = train[columns]
X_test = test[columns]

# Replace all NaN's with 0.
X_train = X_train.fillna(0)
y_train = y_train.fillna(0)
X_test = X_test.fillna(0)
y_test = y_test.fillna(0)

# Create and train the classifiers.
print('Training random forest classifier...')
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, y_train)

print('Training xgboost classifier...')
boost = XGBClassifier()
boost.fit(X_train, y_train)

print('Training neural network classifier...')
neural_network = MLPClassifier(hidden_layer_sizes=(100, 100, 100))
neural_network.fit(X_train, y_train)

# Make predictions and print the accuracy.
def predict(classifier):

    print(type(classifier))

    prediction = classifier.predict(X_test)
    prediction = [round(value) for value in prediction]

    print(f'prediction: {prediction}')
    print(f'reality:    {y_test.tolist()}')

    accuracy = accuracy_score(y_test, prediction)
    print(f'accuracy:   {accuracy * 100}%\n')

predict(random_forest)
predict(boost)
predict(neural_network)