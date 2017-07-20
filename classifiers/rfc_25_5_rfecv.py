import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier as VC
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import GridSearchCV as GS
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import AdaBoostClassifier
from itertools import combinations
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.decomposition import PCA, IncrementalPCA as IPCA
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, Normalizer
from mlxtend.classifier import EnsembleVoteClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import manifold
from operator import mul
from random import shuffle
import sys
sys.path.append('../functions')
from classifier_functions import *
from numerai_functions import *

def main():
    script_name = os.path.basename(__file__).split('.')[0]
    seed = 9000
    np.random.seed(seed)
    final = sys.argv[1]
    print("{} - Loading data...".format(script_name))
    training_data, validation_data, test_data, tournament_data, tournament_ids, targets_train, targets_validation = get_numerai_data('../data/')
    features = [f for f in list(training_data) if "feature" in f]

    final_string = ""
    if final == "1":
        training_data = pd.concat([training_data,validation_data])
        targets_train = pd.concat([targets_train,targets_validation])
        final_string = "_final"

    comb = list(combinations(features,2))
    shuffle(comb)
    for cmb in comb:
        for function in ['addition','subtraction','multiplication','division']:
            if function == 'addition':
                new_feature_name = "_+_".join(cmb)
                training_data[new_feature_name] = training_data[cmb[0]] + training_data[cmb[1]]
                validation_data[new_feature_name] = validation_data[cmb[0]] + validation_data[cmb[1]]
                test_data[new_feature_name] = test_data[cmb[0]] + test_data[cmb[1]]
                tournament_data[new_feature_name] = tournament_data[cmb[0]] + tournament_data[cmb[1]]
            elif function == 'subtraction':
                new_feature_name = "_-_".join(cmb)
                training_data[new_feature_name] = training_data[cmb[0]] - training_data[cmb[1]]
                validation_data[new_feature_name] = validation_data[cmb[0]] - validation_data[cmb[1]]
                test_data[new_feature_name] = test_data[cmb[0]] - test_data[cmb[1]]
                tournament_data[new_feature_name] = tournament_data[cmb[0]] - tournament_data[cmb[1]]
            elif function == 'multiplication':
                new_feature_name = "_*_".join(cmb)
                training_data[new_feature_name] = training_data[cmb[0]] * training_data[cmb[1]]
                validation_data[new_feature_name] = validation_data[cmb[0]] * validation_data[cmb[1]]
                test_data[new_feature_name] = test_data[cmb[0]] * test_data[cmb[1]]
                tournament_data[new_feature_name] = tournament_data[cmb[0]] * tournament_data[cmb[1]]
            elif function == 'division':
                new_feature_name = "_/_".join(cmb)
                training_data[new_feature_name] = training_data[cmb[0]] / (training_data[cmb[1]]+0.01)
                validation_data[new_feature_name] = validation_data[cmb[0]] / (validation_data[cmb[1]]+0.01)
                test_data[new_feature_name] = test_data[cmb[0]] / (test_data[cmb[1]]+0.01)
                tournament_data[new_feature_name] = tournament_data[cmb[0]] / (tournament_data[cmb[1]]+0.01)

    features = [f for f in list(training_data) if "feature" in f]

    features_train = training_data[features]
    features_validation = validation_data[features]
    features_test = test_data[features]
    features_tournament = tournament_data[features]
    ### create classifiers ###
    model = Pipeline([
        ('scale', StandardScaler()),
        ('rfc', RFC())
    ])
    parameters = {
       'rfc__n_estimators': [5,10,25,50,100],
       'rfc__max_depth': [3,5,7,10]
    }

    model = GS(estimator=model, param_grid=parameters)
    ### train classifiers ###
    print("{} - Training...".format(script_name))
    model.fit(features_train, targets_train)
    logloss = calculate_accuracy(script_name, model, features_validation, targets_validation)


if __name__ == '__main__':
    main()
