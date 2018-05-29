from numpy import loadtxt
import pandas as pd

def load_data():

    data = pd.read_csv('data/races.csv')
    np_data = data.values

    training = np_data[:-1]
    last = np_data[-1:]

    print(data.shape)







print('load data...')
load_data()

