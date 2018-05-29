from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
import pandas as pd

print('predict circuit based on year and round')

data = pd.read_csv('data/races.csv')
feature_cols = ['year', 'round']

# X = data.loc[:data.last_valid_index() - 1, feature_cols]
X = data.loc[:, feature_cols]
y = data.name
# print(X)
# print(y)

classifier = DecisionTreeClassifier()
classifier.fit(X, y)

d = {'year': [2019, 2019, 2019, 2019, 2019],
     'round': [1, 2, 3, 4, 5]}
df = pd.DataFrame(data=d)
prediction = classifier.predict(df)
print(prediction)