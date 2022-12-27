import pickle


def load_model():
    model = pickle.load(open('model/NearestNeighbors.pkl', 'rb'))
    return model
