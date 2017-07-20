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
    # for cmb in combs:
    #     new_feature_name = "-".join(cmb)
    #     new_feature_training = training_data[cmb[0]] - training_data[cmb[1]]
    #     new_feature_validation = validation_data[cmb[0]] - validation_data[cmb[1]]
    #     new_feature_test = test_data[cmb[0]] - test_data[cmb[1]]
    #     new_feature_tournament = tournament_data[cmb[0]] - tournament_data[cmb[1]]
    #     training_data[new_feature_name] = new_feature_training
    #     validation_data[new_feature_name] = new_feature_validation
    #     test_data[new_feature_name] = new_feature_test
    #     tournament_data[new_feature_name] = new_feature_tournament
    # features = [f for f in list(training_data) if "feature" in f]

    # comb = [('feature6', 'feature18'), ('feature10', 'feature11'), ('feature12', 'feature18'), ('feature14', 'feature18'), ('feature4', 'feature12')]
    # for c in comb:
    #     c_name = "{}_-_{}".format(c[0], c[1])
    #     training_data[c_name] = training_data[c[0]] - training_data[c[1]]
    #     test_data[c_name] = test_data[c[0]] - test_data[c[1]]
    #     validation_data[c_name] = validation_data[c[0]] - validation_data[c[1]]
    #     tournament_data[c_name] = tournament_data[c[0]] - tournament_data[c[1]]

    features = [f for f in list(training_data) if "feature" in f]

    features_train = training_data[features]
    features_validation = validation_data[features]
    features_test = test_data[features]
    features_tournament = tournament_data[features]

    ### create classifiers ###
    model_1 = Pipeline([
        ('scaler', StandardScaler()),
        ('mlp', MLPClassifier(hidden_layer_sizes=(3,), solver='lbfgs', alpha=0.001))
    ])

    model_2 = Pipeline([
        ('scaler', StandardScaler()),
        ('mlp', MLPClassifier(hidden_layer_sizes=(3,), solver='lbfgs', alpha=0.001))
    ])

    model_3 = Pipeline([
        ('scaler', StandardScaler()),
        ('mlp', MLPClassifier(hidden_layer_sizes=(3,), solver='lbfgs', alpha=0.001))
    ])

    model_4 = Pipeline([
        ('scaler', StandardScaler()),
        ('mlp', MLPClassifier(hidden_layer_sizes=(3,), solver='lbfgs', alpha=0.001))
    ])

    model_5 = Pipeline([
        ('scaler', StandardScaler()),
        ('mlp', MLPClassifier(hidden_layer_sizes=(3,), solver='lbfgs', alpha=0.001))
    ])

    model_6 = Pipeline([
        ('scaler', StandardScaler()),
        ('mlp', MLPClassifier(hidden_layer_sizes=(3,), solver='lbfgs', alpha=0.001))
    ])

    model_7 = Pipeline([
        ('scaler', StandardScaler()),
        ('mlp', MLPClassifier(hidden_layer_sizes=(3,), solver='lbfgs', alpha=0.001))
    ])

    model_8 = Pipeline([
        ('scaler', StandardScaler()),
        ('mlp', MLPClassifier(hidden_layer_sizes=(3,), solver='lbfgs', alpha=0.001))
    ])

    model_9 = Pipeline([
        ('scaler', StandardScaler()),
        ('mlp', MLPClassifier(hidden_layer_sizes=(3,), solver='lbfgs', alpha=0.001))
    ])

    model_10 = Pipeline([
        ('scaler', StandardScaler()),
        ('mlp', MLPClassifier(hidden_layer_sizes=(3,), solver='lbfgs', alpha=0.001))
    ])

    model = EnsembleVoteClassifier(clfs=[model_1, model_2, model_3, model_4, model_5, model_6, model_7, model_8, model_9, model_10], voting='soft', weights=[1,1,1,1,1,1,1,1,1,1])

    ### train classifiers ###
    print("{} - Training...".format(script_name))
    model.fit(features_train, targets_train)
    calculate_accuracy(script_name, model, features_validation, targets_validation)

    prob_predictions_validation = model.predict_proba(features_validation)
    prob_predicitons_validation = prob_predictions_validation[:, 1]
    prob_predicitons_validation = pd.DataFrame(data={'probability':prob_predicitons_validation})
    joined = pd.DataFrame(targets_validation).join(prob_predicitons_validation)
    print("{} - Writing Validation Results...".format(script_name))
    joined.to_csv("../predictions/{}_validation{}.csv".format(script_name, final_string), index=False)

    prob_predictions_tournament = model.predict_proba(features_tournament)
    results = prob_predictions_tournament[:, 1]
    results_df = pd.DataFrame(data={'probability':results})
    joined = pd.DataFrame(tournament_ids).join(results_df)
    print("{} - Writing...".format(script_name))
    joined.to_csv("../predictions/{}{}.csv".format(script_name, final_string), index=False)

if __name__ == '__main__':
    main()
