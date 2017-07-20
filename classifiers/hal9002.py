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

    if final == "1":
        training_data = pd.concat([training_data,validation_data])
        targets_train = pd.concat([targets_train,targets_validation])

    ### Feature Creation ###
    print("{} - Creating New Features...".format(script_name))

    # training_data['feature4_-_feature21'] = training_data['feature4'] - training_data['feature21']
    # test_data['feature4_-_feature21'] = test_data['feature4'] - test_data['feature21']
    # validation_data['feature4_-_feature21'] = validation_data['feature4'] - validation_data['feature21']
    # tournament_data['feature4_-_feature21'] = tournament_data['feature4'] - tournament_data['feature21']

    # training_data['feature5_-_feature17'] = training_data['feature5'] - training_data['feature17']
    # test_data['feature5_-_feature17'] = test_data['feature5'] - test_data['feature17']
    # validation_data['feature5_-_feature17'] = validation_data['feature5'] - validation_data['feature17']
    # tournament_data['feature5_-_feature17'] = tournament_data['feature5'] - tournament_data['feature17']

    # training_data['feature10_-_feature19'] = training_data['feature10'] - training_data['feature19']
    # test_data['feature10_-_feature19'] = test_data['feature10'] - test_data['feature19']
    # validation_data['feature10_-_feature19'] = validation_data['feature10'] - validation_data['feature19']
    # tournament_data['feature10_-_feature19'] = tournament_data['feature10'] - tournament_data['feature19']

    training_data['feature16_-_feature18'] = training_data['feature16'] - training_data['feature18']
    test_data['feature16_-_feature18'] = test_data['feature16'] - test_data['feature18']
    validation_data['feature16_-_feature18'] = validation_data['feature16'] - validation_data['feature18']
    tournament_data['feature16_-_feature18'] = tournament_data['feature16'] - tournament_data['feature18']

    features = [f for f in list(training_data) if "feature" in f]
    print(features)
    features_train = training_data[features]
    features_validation = validation_data[features]
    features_test = test_data[features]
    features_tournament = tournament_data[features]

    ### create classifiers ###
    model = make_pipeline(
        StandardScaler(),
        IPCA(),
        GradientBoostingClassifier(n_estimators=25, max_depth=5)
    )

    ### train classifiers ###
    print("{} - Training...".format(script_name))
    model.fit(features_train, targets_train)
    calculate_accuracy(script_name, model, features_validation, targets_validation)

    prob_predictions_test = model.predict_proba(features_tournament)
    results = prob_predictions_test[:, 1]
    results_df = pd.DataFrame(data={'probability':results})
    joined = pd.DataFrame(tournament_ids).join(results_df)
    print("{} - Writing...".format(script_name))
    joined.to_csv("../predictions/{}.csv".format(script_name), index=False)

if __name__ == '__main__':
    main()
