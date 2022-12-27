import argparse
import logging

import numpy as np

from model.model import get_df, my_train_test_split, fit_KNeighborsRegressor, fit_KNeighborsClassifier
from util.utils import load_model

parser = argparse.ArgumentParser(description='CLI for classification.')
parser.add_argument('--prediction_model', dest='model', type=str, nargs=1,
                    help='classification model')
parser.add_argument('--prediction_params', dest='params', type=str, nargs=1,
                    help='input parameters for the model')
args = parser.parse_args()

model = args.model[0]
params = np.reshape(list(map(float, args.params[0].split(','))), (1, -1))

df = get_df()

X_train, X_test, y_train, y_test = my_train_test_split(df)
fit_KNeighborsRegressor(X_train, y_train)
neigh = load_model()
logging.info(f'Accuracy is {neigh.score(X_test, y_test)}')
logging.info(f'Predictions on test are {neigh.predict(X_test)}')
logging.info(f'Target prediction: {neigh.predict(params)}')

fit_KNeighborsClassifier(X_train, y_train)
neigh = load_model()
logging.info(f'Accuracy is {neigh.score(X_test, y_test)}')
logging.info(f'Predictions on test are {neigh.predict(X_test)}')
logging.info(f'Target prediction: {neigh.predict(params)}')
