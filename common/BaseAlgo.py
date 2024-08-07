import pandas as pd
from sklearn import preprocessing

class BaseAlgo():

    def __init__(self):
        pass

    def load_data(self, path):
        df = pd.read_csv(path)#panda datasets
        df.head()
        print("Succesfully loaded {}".format(path))
        return df

    def drop_features(self,df,col,axis=1):
        df = df.drop(col, axis=axis)
        df.head()
        return df

    def normalize(self,X):
        X_transform = preprocessing.StandardScaler().fit_transform(X)
        return X_transform


