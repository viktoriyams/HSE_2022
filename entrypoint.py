import argparse
import logging

import numpy as np

from model.model import get_df, my_train_test_split
from model.KNR import fit_KNeighborsRegressor
from model.KNC import fit_KNeighborsClassifier
from util.utils import load_model_KNC, load_model_KNR

parser = argparse.ArgumentParser(description='CLI for models')
parser.add_argument('--prediction_model', dest='model', type=str, nargs=1,
                    help='classification model')
parser.add_argument('--prediction_params', dest='params', type=str, nargs=1,
                    help='input parameters for the model')
args = parser.parse_args()

model = args.model[0].split(",")
params = np.reshape(list(map(float, args.params[0].split(','))), (1, -1))

df = get_df()
X_train, X_test, y_train, y_test = my_train_test_split(df)

for i in model:
    if i == 'KNeighborsRegressor':
        fit_KNeighborsRegressor(X_train, y_train)
        neigh = load_model_KNR()
        logging.info(f'Accuracy is {neigh.score(X_test, y_test)}')
        logging.info(f'Predictions on test are {neigh.predict(X_test)}')
        logging.info(f'Target prediction: {neigh.predict(params)}')
    elif i == 'KNeighborsClassifier':
        fit_KNeighborsClassifier(X_train, y_train)
        neigh = load_model_KNC()
        logging.info(f'Accuracy is {neigh.score(X_test, y_test)}')
        logging.info(f'Predictions on test are {neigh.predict(X_test)}')
        logging.info(f'Target prediction: {neigh.predict(params)}')
    else:
        logging.info(f'No model with name: {i}')



