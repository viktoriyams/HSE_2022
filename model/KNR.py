import pickle

from sklearn.neighbors import KNeighborsRegressor

from conf.conf import logging

def fit_KNeighborsRegressor(X_train, y_train):
    logging.info('Train KNeighborsRegressor model')
    neigh = KNeighborsRegressor(n_neighbors=2).fit(X_train, y_train)
    f = open('model/conf/KNeighborsRegressor.pkl', 'wb')
    pickle.dump(neigh, f)
    f.close()
    return neigh