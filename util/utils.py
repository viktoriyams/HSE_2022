import pickle


def load_model_KNC():
    model = pickle.load(open('model/conf/KNeighborsClassifier.pkl', 'rb'))
    return model

def load_model_KNR():
    model = pickle.load(open('model/conf/KNeighborsRegressor.pkl', 'rb'))
    return model
