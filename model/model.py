import pickle

from sklearn.model_selection import train_test_split
"""from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import KNeighborsClassifier"""


from conf.conf import logging, settings
from connector.pg_connector import get_data


def my_train_test_split(df):
    logging.info('Extracting X and y')
    X = df.iloc[:, :-1]
    y = df['target']

    logging.info('Splitting X and y for test and train')
    X_train, X_test, y_train, y_test = train_test_split(X,  # independent variables
                                                        y,  # dependent variable
                                                        random_state=3
                                                        )
    return X_train.values, X_test.values, y_train.values, y_test.values

"""
def fit_KNeighborsRegressor(X_train, y_train):
    logging.info('Train KNeighborsRegressor model')
    neigh = KNeighborsRegressor(n_neighbors=2).fit(X_train, y_train)
    f = open('model/NearestNeighbors.pkl', 'wb')
    pickle.dump(neigh, f)
    f.close()
    return neigh

def fit_KNeighborsClassifier(X_train, y_train):
    logging.info('Train KNeighborsClassifier model')
    neigh = KNeighborsClassifier(n_neighbors=2).fit(X_train, y_train)
    f = open('model/NearestNeighbors.pkl', 'wb')
    pickle.dump(neigh, f)
    f.close()
    return neigh"""

def get_df():
    logging.info(f'Extract dataset from {settings.DATA.data_set}')
    df = get_data(settings.DATA.data_set)
    logging.info('Extracted dataset')
    return df
