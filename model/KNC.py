import pickle

from sklearn.neighbors import KNeighborsClassifier

from conf.conf import logging

def fit_KNeighborsClassifier(X_train, y_train):
    logging.info('Train KNeighborsClassifier model')
    neigh = KNeighborsClassifier(n_neighbors=2).fit(X_train, y_train)
    f = open('model/conf/KNeighborsClassifier.pkl', 'wb')
    pickle.dump(neigh, f)
    f.close()
    return neigh