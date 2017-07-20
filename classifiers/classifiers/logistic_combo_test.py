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
from sklearn.pipeline import make_pipeline
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

    ### Feature Creation ###
    print("{} - Creating New Features...".format(script_name))

    ### 2-combo multiplication ###
    # for cmb in combinations(features, 2):
    #     new_feature_name = "-".join(cmb)
    #     new_feature_training = training_data[cmb[0]] * training_data[cmb[1]]
    #     new_feature_validation = validation_data[cmb[0]] * validation_data[cmb[1]]
    #     new_feature_test = test_data[cmb[0]] * test_data[cmb[1]]
    #     new_feature_tournament = tournament_data[cmb[0]] * tournament_data[cmb[1]]
    #     training_data[new_feature_name] = new_feature_training
    #     validation_data[new_feature_name] = new_feature_validation
    #     test_data[new_feature_name] = new_feature_test
    #     tournament_data[new_feature_name] = new_feature_tournament
    # features = [f for f in list(training_data) if "feature" in f]
    saved_features = []
    benchmark = 0.692451
    result_file = '../logistic_combe_test.txt'
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

    new_features = [f for f in list(training_data) if "_feature" in f]
    shuffle(new_features)
    for nf in new_features:
        print nf
        test_features = list(features)
        test_features.append(nf)
        features_train = training_data[test_features]
        features_validation = validation_data[test_features]
        features_test = test_data[test_features]
        features_tournament = tournament_data[test_features]
        ### create classifiers ###
        model = make_pipeline(
            # PolynomialFeatures(),
            # Normalizer(),
            # RFECV(estimator=LogisticRegression(C=0.3), step=0.2),
            IPCA(),
            LogisticRegression(C=1)
        )

        ### train classifiers ###
        print("{} - Training...".format(script_name))
        model.fit(features_train, targets_train)
        logloss = calculate_accuracy(script_name, model, features_validation, targets_validation)
        if logloss < benchmark:
            features.append(nf)
            print "{} - Reduced Logloss to {}".format(nf, logloss)
            benchmark = logloss
            with open(result_file, 'a') as f:
                f.write('%s : %s\n' % (features, logloss))

if __name__ == '__main__':
    main()
