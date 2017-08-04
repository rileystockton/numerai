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
from sklearn import manifold, linear_model
import matplotlib.pyplot as plt
import seaborn as sns
from operator import mul
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

    # features = ['feature11_/_feature16', 'feature16_+_feature17', 'feature2_+_feature21', 'feature4_-_feature16', 'feature14_/_feature19']
    features = ['feature11_/_feature16', 'feature16_+_feature17', 'feature2_+_feature21', 'feature11_+_feature17', 'feature4_-_feature16', 'feature6_/_feature21', 'feature15_*_feature19', 'feature15_+_feature19', 'feature11_*_feature20', 'feature4_-_feature11', 'feature4_/_feature11', 'feature10_/_feature11']
    features_train = training_data[features]
    features_validation = validation_data[features]
    features_test = test_data[features]
    features_tournament = tournament_data[features]

    ### create classifiers ###
    model = make_pipeline(
        linear_model.Ridge(alpha=0.8)
    )

    ### train classifiers ###
    print("{} - Training...".format(script_name))
    model.fit(features_train, targets_train)
    logloss = cross_val_score(model, features_train, targets_train, cv=3, scoring='neg_mean_squared_error')
    print("%s - Logloss: %0.6f (+/- %0.6f)" % (script_name, logloss.mean(), logloss.std() * 2))

    prob_predictions_validation = model.predict(features_validation)
    prob_predictions_validation = pd.DataFrame(data={'probability':prob_predictions_validation})
    joined = pd.DataFrame(targets_validation).join(prob_predictions_validation)
    print("{} - Writing Validation Results...".format(script_name))
    joined.to_csv("../predictions/{}_validation{}.csv".format(script_name, final_string), index=False)

    results = model.predict(features_tournament)
    results_df = pd.DataFrame(data={'probability':results})
    joined = pd.DataFrame(tournament_ids).join(results_df)
    print("{} - Writing...".format(script_name))
    joined.to_csv("../predictions/{}{}.csv".format(script_name, final_string), index=False)

if __name__ == '__main__':
    main()
