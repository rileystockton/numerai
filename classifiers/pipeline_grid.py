import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.ensemble import GradientBoostingClassifier
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

    if final == "1":
        training_data = pd.concat([training_data,validation_data])
        targets_train = pd.concat([targets_train,targets_validation])

    ### Get Feature Sets ###
    features = [f for f in list(training_data) if "feature" in f]
    features_train = training_data[features]
    features_validation = validation_data[features]
    features_test = test_data[features]
    features_tournament = tournament_data[features]

    ### Scale Inputs ###
    scaler = StandardScaler()
    scaler.fit(features_train)
    features_train = scaler.transform(features_train)
    features_validation = scaler.transform(features_validation)
    features_test = scaler.transform(features_test)
    features_tournament = scaler.transform(features_tournament)


    ### create classifiers ###
    #.69196
    model_1 = make_pipeline(
        IPCA(),
        GradientBoostingClassifier(n_estimators=25, max_depth=5)
    )

    # model_2 = make_pipeline(
    #     StandardScaler(),
    #     IPCA(),
    #     GradientBoostingClassifier(n_estimators=25)
    # )
    # #.69248
    # model_3 = make_pipeline(
    #     Normalizer(),
    #     RFECV(estimator=LogisticRegression(C=0.3), step=1),
    #     LogisticRegression(C=0.3)
    # )
    # #.69217
    # model_4 = make_pipeline(
    #     # PolynomialFeatures(),
    #     StandardScaler(),
    #     IPCA(),
    #     RFC(n_estimators=25, max_depth=7)
    # )
    #
    model_5 = Pipeline([
        ('mlp', MLPClassifier())
    ])

    ### create classifiers ###
    model_6 = Pipeline([
        # PolynomialFeatures(),
        # Normalizer(),
        ('ipca', IPCA()),
        ('rfc', RFC(n_estimators=25))
    ])

    parameters = {
       'rfc__max_depth': [3,4,5],
       'ipca__n_components': [3,4,5]
    }

    model = GS(estimator=model_6, param_grid=parameters)

    ### train classifiers ###
    print("{} - Training...".format(script_name))
    model.fit(features_train, targets_train)
    logloss = cross_val_score(model, features_train, targets_train, cv=3, scoring='neg_log_loss')
    print("%s - Logloss: %0.6f (+/- %0.6f)" % (script_name, logloss.mean(), logloss.std() * 2))
    print("{} - Best Hyperparameters: {}".format(script_name,str(model.best_params_)))
    prob_predictions_test = model.predict_proba(features_tournament)
    results = prob_predictions_test[:, 1]
    results_df = pd.DataFrame(data={'probability':results})
    joined = pd.DataFrame(tournament_ids).join(results_df)
    print("{} - Writing...".format(script_name))
    joined.to_csv("../predictions/{}.csv".format(script_name), index=False)

if __name__ == '__main__':
    main()
