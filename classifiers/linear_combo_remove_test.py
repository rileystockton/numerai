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
    benchmark = 0.24979109839
    result_file = '../linear_combo_remove_test.txt'
    for cmb in combinations(features,2):
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

    features = ['feature1', 'feature2', 'feature3', 'feature4', 'feature5', 'feature6', 'feature7', 'feature8', 'feature9', 'feature10', 'feature11', 'feature12', 'feature13', 'feature14', 'feature15', 'feature16', 'feature17', 'feature18', 'feature19', 'feature20', 'feature21', 'feature2_*_feature17', 'feature13_/_feature16', 'feature2_/_feature6', 'feature4_/_feature17', 'feature8_/_feature13', 'feature8_*_feature19', 'feature5_/_feature11', 'feature1_*_feature19', 'feature1_+_feature7', 'feature1_-_feature19', 'feature5_-_feature21', 'feature14_-_feature18', 'feature20_*_feature21', 'feature7_-_feature18', 'feature5_+_feature15', 'feature1_-_feature8', 'feature8_+_feature20', 'feature4_+_feature19', 'feature5_-_feature19', 'feature12_-_feature19', 'feature3_+_feature4', 'feature5_+_feature7', 'feature14_+_feature15', 'feature18_+_feature19', 'feature5_+_feature19', 'feature5_/_feature10', 'feature2_+_feature5', 'feature8_-_feature13', 'feature11_+_feature14', 'feature8_-_feature15', 'feature13_+_feature19', 'feature1_*_feature7', 'feature10_/_feature16', 'feature14_*_feature20', 'feature2_+_feature11', 'feature11_/_feature19', 'feature3_-_feature16', 'feature3_-_feature14', 'feature8_+_feature17', 'feature7_-_feature21', 'feature1_+_feature16', 'feature3_-_feature18', 'feature2_-_feature3', 'feature1_*_feature6', 'feature7_-_feature8', 'feature7_-_feature15', 'feature14_-_feature17', 'feature17_-_feature18', 'feature4_-_feature8', 'feature11_/_feature17', 'feature11_-_feature15', 'feature3_-_feature21', 'feature5_+_feature21', 'feature7_+_feature14', 'feature12_+_feature14', 'feature8_+_feature21', 'feature8_+_feature13', 'feature15_+_feature21', 'feature1_+_feature21', 'feature18_-_feature21', 'feature5_+_feature17']

    shuffle(features)
    for ft in features:
        print ft
        test_features = list(features)
        test_features.remove(ft)
        features_train = training_data[test_features]
        features_validation = validation_data[test_features]
        features_test = test_data[test_features]
        features_tournament = tournament_data[test_features]
        ### create classifiers ###
        model = make_pipeline(
            linear_model.Ridge(alpha=0.8)
        )

        ### train classifiers ###
        print("{} - Training...".format(script_name))
        model.fit(features_train, targets_train)
        score = cross_val_score(model, features_train, targets_train, cv=5, scoring='neg_mean_squared_error')
        mse = -score.mean()
        if mse < benchmark:
            features.remove(ft)
            print "Removing {} - Reduced Logloss to {}".format(ft, mse)
            benchmark = mse
            with open(result_file, 'a') as f:
                f.write('%s : %s\n' % (features, mse))

if __name__ == '__main__':
    main()
