from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
import pandas as pd

print('predict circuit based on year and round')

df = pd.read_csv('data/races.csv')
feature_cols = ['year', 'date']

le = LabelEncoder()
le.fit(df.date.unique())
df.date = le.transform(df.date)

# X = df.loc[:df.last_valid_index() - 1, feature_cols]
X = df.loc[:, feature_cols]
y = df.name
# print(X)
# print(y)







# le.fit(['one', 'two', 'two', 'three'])
# print(le.classes_)
# print(le.transform(['one', 'two', 'two', 'three']))
# print(le.inverse_transform(le.transform(['one', 'two', 'two', 'three'])))

print(X)


inv = le.inverse_transform([1, 2, 3, 4, 5])
print(inv)





classifier = DecisionTreeClassifier()
classifier.fit(X, y)

d = {'year': [2018, 2018, 2018, 2018, 2018],
     'date': [1, 2, 3, 4, 5]}
df = pd.DataFrame(data=d)
prediction = classifier.predict(df)
print(prediction)