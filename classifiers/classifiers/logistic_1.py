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

    features = ['feature1', 'feature2', 'feature3', 'feature4', 'feature5', 'feature6', 'feature7', 'feature8', 'feature9', 'feature10', 'feature11', 'feature12', 'feature13', 'feature14', 'feature15', 'feature16', 'feature17', 'feature18', 'feature19', 'feature20', 'feature21', 'feature11_-_feature16', 'feature5_/_feature15', 'feature3_-_feature11', 'feature1_*_feature6', 'feature4_+_feature8', 'feature4_-_feature12', 'feature1_+_feature4', 'feature7_*_feature16', 'feature6_+_feature18', 'feature15_/_feature21', 'feature14_/_feature20', 'feature1_+_feature19', 'feature13_-_feature21', 'feature3_-_feature5', 'feature4_/_feature19', 'feature4_/_feature5', 'feature4_+_feature17', 'feature13_+_feature14', 'feature3_+_feature4', 'feature2_/_feature6', 'feature2_-_feature16', 'feature8_-_feature21', 'feature5_+_feature8', 'feature6_+_feature14', 'feature4_-_feature20', 'feature4_-_feature15', 'feature4_+_feature14', 'feature6_*_feature12', 'feature6_*_feature10', 'feature12_*_feature19', 'feature6_+_feature21', 'feature4_*_feature19', 'feature5_-_feature20', 'feature2_+_feature5', 'feature2_+_feature6', 'feature1_-_feature20', 'feature7_*_feature15', 'feature8_*_feature19', 'feature6_-_feature7', 'feature4_+_feature12', 'feature5_+_feature20', 'feature9_/_feature16', 'feature15_+_feature21', 'feature16_/_feature19', 'feature9_*_feature16', 'feature7_+_feature11', 'feature7_+_feature17', 'feature8_/_feature21', 'feature6_*_feature13', 'feature7_+_feature15', 'feature5_+_feature18', 'feature7_+_feature12', 'feature3_/_feature18', 'feature5_-_feature17', 'feature6_*_feature9', 'feature5_-_feature6', 'feature4_+_feature7', 'feature6_+_feature10', 'feature6_*_feature16', 'feature6_-_feature11', 'feature9_-_feature19', 'feature2_-_feature6', 'feature7_-_feature18', 'feature1_-_feature21', 'feature2_/_feature11', 'feature6_*_feature19', 'feature4_-_feature5', 'feature1_/_feature6', 'feature7_/_feature15', 'feature7_-_feature17', 'feature8_-_feature17', 'feature8_+_feature18', 'feature6_-_feature9', 'feature1_-_feature6', 'feature2_/_feature4', 'feature5_-_feature12', 'feature5_*_feature15', 'feature2_+_feature17', 'feature9_-_feature20', 'feature13_-_feature14', 'feature6_*_feature8', 'feature2_+_feature21', 'feature9_+_feature15', 'feature3_+_feature17', 'feature5_/_feature13', 'feature2_-_feature3', 'feature6_-_feature15', 'feature9_+_feature11', 'feature18_+_feature20', 'feature5_/_feature7', 'feature7_/_feature11', 'feature5_-_feature9', 'feature18_+_feature19', 'feature5_/_feature10', 'feature1_+_feature11', 'feature6_+_feature13', 'feature13_-_feature20', 'feature7_-_feature13', 'feature2_*_feature10', 'feature15_+_feature18', 'feature12_/_feature14', 'feature5_+_feature13', 'feature2_-_feature20', 'feature10_+_feature20', 'feature19_+_feature21', 'feature9_-_feature14', 'feature5_+_feature17', 'feature8_+_feature21', 'feature9_+_feature19', 'feature6_*_feature20', 'feature9_-_feature13', 'feature17_-_feature19', 'feature9_-_feature18', 'feature6_-_feature14', 'feature9_+_feature12', 'feature16_-_feature21', 'feature2_+_feature9', 'feature7_*_feature19', 'feature10_/_feature11', 'feature9_*_feature14', 'feature13_-_feature16', 'feature9_-_feature11', 'feature5_-_feature7', 'feature2_-_feature9', 'feature9_+_feature10', 'feature2_+_feature15', 'feature10_-_feature18', 'feature1_/_feature4', 'feature7_+_feature19', 'feature2_+_feature13', 'feature13_+_feature19', 'feature5_/_feature6', 'feature3_/_feature7', 'feature7_*_feature8', 'feature4_+_feature13', 'feature12_+_feature20', 'feature1_/_feature11', 'feature6_-_feature17', 'feature11_-_feature14', 'feature3_-_feature15', 'feature7_-_feature12']

    features_train = training_data[features]
    features_validation = validation_data[features]
    features_test = test_data[features]
    features_tournament = tournament_data[features]

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
    calculate_accuracy(script_name, model, features_validation, targets_validation)

    prob_predictions_validation = model.predict_proba(features_validation)
    prob_predicitons_validation = prob_predictions_validation[:, 1]
    prob_predicitons_validation = pd.DataFrame(data={'probability':prob_predicitons_validation})
    joined = pd.DataFrame(targets_validation).join(prob_predicitons_validation)
    print("{} - Writing Validation Results...".format(script_name))
    joined.to_csv("../predictions/{}_validation.csv".format(script_name), index=False)

    prob_predictions_tournament = model.predict_proba(features_tournament)
    results = prob_predictions_tournament[:, 1]
    results_df = pd.DataFrame(data={'probability':results})
    joined = pd.DataFrame(tournament_ids).join(results_df)
    print("{} - Writing...".format(script_name))
    joined.to_csv("../predictions/{}{}.csv".format(script_name, final_string), index=False)

if __name__ == '__main__':
    main()
