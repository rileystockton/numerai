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

    features_train = training_data[features]
    features_validation = validation_data[features]
    features_test = test_data[features]
    features_tournament = tournament_data[features]

    ### Remove Outliers ###
    features_train, targets_train = tukey_pca_outlier_removal(features_train, targets_train)

    ### create classifiers ###
    model = make_pipeline(
        LogisticRegression(C=0.3)
    )

    ### train classifiers ###
    print("{} - Training...".format(script_name))
    model.fit(features_train, targets_train)
    logloss = cross_val_score(model, features_train, targets_train, cv=3, scoring='neg_log_loss')
    print("%s - Logloss: %0.6f (+/- %0.6f)" % (script_name, logloss.mean(), logloss.std() * 2))

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
