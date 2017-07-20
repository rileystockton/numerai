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
    # features_plus_era = features + ['era']
    # training_data[features] = training_data[features_plus_era].groupby('era').transform(lambda x: (x - x.mean()))
    # validation_data[features] = validation_data[features_plus_era].groupby('era').transform(lambda x: (x - x.mean()))
    # test_data[features] = test_data[features_plus_era].groupby('era').transform(lambda x: (x - x.mean()))

    ### Feature Creation ###
    print("{} - Creating New Features...".format(script_name))

    # cor = training_data.corr()['target']
    # positive_correlators = [x for x in features if cor[x]>0.01][:4]
    # print(positive_correlators)
    # negative_correlators = [x for x in features if cor[x]<-0.01][:4]
    # print(negative_correlators)
    # for i in range(len(positive_correlators)-1):
    #   for cmb in combinations(positive_correlators, i+2):
    #       new_feature_training, new_feature_validation, new_feature_test, new_feature_tournament = 1, 1, 1, 1
    #       new_feature_name = "_x_".join(cmb)
    #       print(new_feature_name)
    #       for c in cmb:
    #           new_feature_training = new_feature_training * training_data[c]
    #           new_feature_validation = new_feature_validation * validation_data[c]
    #           new_feature_test = new_feature_test * test_data[c]
    #           new_feature_tournament = new_feature_tournament * tournament_data[c]
    #       training_data[new_feature_name] = new_feature_training
    #       validation_data[new_feature_name] = new_feature_validation
    #       test_data[new_feature_name] = new_feature_test
    #       tournament_data[new_feature_name] = new_feature_tournament
    # for i in range(len(negative_correlators)-1):
    #   for cmb in combinations(negative_correlators, i+2):
    #       new_feature_training, new_feature_validation, new_feature_test, new_feature_tournament = 1, 1, 1, 1
    #       new_feature_name = "_x_".join(cmb)
    #       print(new_feature_name)
    #       for c in cmb:
    #           new_feature_training = new_feature_training * training_data[c]
    #           new_feature_validation = new_feature_validation * validation_data[c]
    #           new_feature_test = new_feature_test * test_data[c]
    #           new_feature_tournament = new_feature_tournament * tournament_data[c]
    #       training_data[new_feature_name] = new_feature_training
    #       validation_data[new_feature_name] = new_feature_validation
    #       test_data[new_feature_name] = new_feature_test
    #       tournament_data[new_feature_name] = new_feature_tournament

    # cor = training_data.corr()['target']
    # positive_correlators = [x for x in features if cor[x]>0.01][:4]
    # print(positive_correlators)
    # negative_correlators = [x for x in features if cor[x]<-0.01][:4]
    # print(negative_correlators)
    # for i in range(len(positive_correlators)-1):
    #   for cmb in combinations(positive_correlators, i+2):
    #       new_feature_training, new_feature_validation, new_feature_test, new_feature_tournament = 0, 0, 0, 0
    #       new_feature_name = "_+_".join(cmb)
    #       print(new_feature_name)
    #       for c in cmb:
    #           new_feature_training = new_feature_training + training_data[c]
    #           new_feature_validation = new_feature_validation + validation_data[c]
    #           new_feature_test = new_feature_test + test_data[c]
    #           new_feature_tournament = new_feature_tournament + tournament_data[c]
    #       training_data[new_feature_name] = new_feature_training
    #       validation_data[new_feature_name] = new_feature_validation
    #       test_data[new_feature_name] = new_feature_test
    #       tournament_data[new_feature_name] = new_feature_tournament
    # for i in range(len(negative_correlators)-1):
    #   for cmb in combinations(negative_correlators, i+2):
    #       new_feature_training, new_feature_validation, new_feature_test, new_feature_tournament = 0, 0, 0, 0
    #       new_feature_name = "_+_".join(cmb)
    #       print(new_feature_name)
    #       for c in cmb:
    #           new_feature_training = new_feature_training + training_data[c]
    #           new_feature_validation = new_feature_validation + validation_data[c]
    #           new_feature_test = new_feature_test + test_data[c]
    #           new_feature_tournament = new_feature_tournament + tournament_data[c]
    #       training_data[new_feature_name] = new_feature_training
    #       validation_data[new_feature_name] = new_feature_validation
    #       test_data[new_feature_name] = new_feature_test
    #       tournament_data[new_feature_name] = new_feature_tournament

    # 2-combo multiplication ###
    # for cmb in combinations(features, 2):
    #     new_feature_training, new_feature_validation, new_feature_test, new_feature_tournament = 1, 1, 1, 1
    #     new_feature_name = "_x_".join(cmb)
    #     for c in cmb:
    #         new_feature_training = new_feature_training * training_data[c]
    #         new_feature_validation = new_feature_validation * validation_data[c]
    #         new_feature_test = new_feature_test * test_data[c]
    #         new_feature_tournament = new_feature_tournament * tournament_data[c]
    #     training_data[new_feature_name] = new_feature_training
    #     validation_data[new_feature_name] = new_feature_validation
    #     test_data[new_feature_name] = new_feature_test
    #     tournament_data[new_feature_name] = new_feature_tournament
    # features = [f for f in list(training_data) if "feature" in f]

    # mult_combos = [('3','11')]
    # #
    # for cmb in mult_combos:
    #     print(cmb)
    #     feature_1 = 'feature' + cmb[0]
    #     feature_2 = 'feature' + cmb[1]
    #     new_feature_name = feature_1 + '_x_' + feature_2
    #     new_feature_training = training_data[feature_1] * training_data[feature_2]
    #     new_feature_validation = validation_data[feature_1] * validation_data[feature_2]
    #     new_feature_test = test_data[feature_1] * test_data[feature_2]
    #     new_feature_tournament = tournament_data[feature_1] * tournament_data[feature_2]
    #     training_data[new_feature_name] = new_feature_training
    #     validation_data[new_feature_name] = new_feature_validation
    #     test_data[new_feature_name] = new_feature_test
    #     tournament_data[new_feature_name] = new_feature_tournament
    # # print(training_data.head(5))
    # features = [f for f in list(training_data) if "feature" in f]
    # features.remove('feature2')
    # features.remove('feature10')
    # features.remove('feature13')


    ## reduce feature set ###
    # rfe_estimator = RFC(n_estimators=25, max_depth=5)
    # rfecv = RFECV(estimator=rfe_estimator, step=0.5)
    # scaler = StandardScaler()
    # print("{} - Scaling Feature Set...".format(script_name))
    # features_train = scaler.fit_transform(training_data[features])
    # print("{} - Reducing Feature Set...".format(script_name))
    # rfecv.fit(features_train, targets_train)
    # print(rfecv.ranking_)
    # used_features = rfecv.get_support(indices=True).tolist()
    # reduced_features = [features[i] for i in used_features]
    # print("{} - {} Features Chosen".format(script_name, len(reduced_features)))
    # print(reduced_features)
    # features = reduced_features
    # features = ['feature4', 'feature9', 'feature20', 'feature56', 'feature59', 'feature70', 'feature75', 'feature83', 'feature96', 'feature111', 'feature113', 'feature116', 'feature117', 'feature124', 'feature125', 'feature126', 'feature145', 'feature183', 'feature185', 'feature192', 'feature193', 'feature200', 'feature202', 'feature204', 'feature233', 'feature235', 'feature236', 'feature247', 'feature249', 'feature251']

    ### create polynomial features ###
    # poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)
    # features_train = poly.fit_transform(training_data[features])
    # features_train = pd.DataFrame(features_train)
    # features_validation = poly.fit_transform(validation_data[features])
    # features_validation = pd.DataFrame(features_validation)
    # features_test = poly.fit_transform(test_data[features])
    # features_test = pd.DataFrame(features_test)
    # features_tournament = poly.fit_transform(tournament_data[features])
    # features_tournament = pd.DataFrame(features_tournament)
    # ### rename all new features ###
    # features_train.columns = ['feature'+str(x) for x in range(len(features_train.columns))]
    # features_validation.columns = ['feature'+str(x) for x in range(len(features_validation.columns))]
    # features_test.columns = ['feature'+str(x) for x in range(len(features_test.columns))]
    # features_tournament.columns = ['feature'+str(x) for x in range(len(features_tournament.columns))]
    # features = [f for f in list(features_train) if "feature" in f]

    # reduce feature set ###
    # rfe_estimator = LogisticRegression()
    # rfecv = RFECV(estimator=rfe_estimator, step=0.1)
    # scaler = StandardScaler()
    # print("{} - Scaling Feature Set...".format(script_name))
    # features_train = scaler.fit_transform(training_data[features])
    # print("{} - Reducing Feature Set...".format(script_name))
    # rfecv.fit(features_train, targets_train)
    # used_features = rfecv.get_support(indices=True).tolist()
    # print(features)
    # reduced_features = [features[i] for i in used_features]
    # print(reduced_features)
    # print("{} - {} Features Chosen".format(script_name, len(reduced_features)))
    # print(reduced_features)
    # features = reduced_features
    # features = ['feature4', 'feature9', 'feature20', 'feature56', 'feature59', 'feature70', 'feature75', 'feature83', 'feature96', 'feature111', 'feature113', 'feature116', 'feature117', 'feature124', 'feature125', 'feature126', 'feature145', 'feature183', 'feature185', 'feature192', 'feature193', 'feature200', 'feature202', 'feature204', 'feature233', 'feature235', 'feature236', 'feature247', 'feature249', 'feature251']

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

    model_2 = make_pipeline(
        IPCA(),
        GradientBoostingClassifier(n_estimators=25)
    )
    #.69248
    model_3 = make_pipeline(
        Normalizer(),
        # RFECV(estimator=LogisticRegression(C=0.3), step=1),
        LogisticRegression(C=0.3)
    )
    #.69217
    model_4 = make_pipeline(
        # PolynomialFeatures(),
        IPCA(),
        RFC(n_estimators=25, max_depth=7)
    )

    model_5 = make_pipeline(
        IPCA(),
        MLPClassifier(hidden_layer_sizes=(3,10), solver='lbfgs', alpha=0.0001)
    )

    model_6 = make_pipeline(
        IPCA(),
        MLPClassifier(hidden_layer_sizes=(3,10), solver='lbfgs', alpha=0.00001)
    )

    model_7 = make_pipeline(
        IPCA(),
        MLPClassifier(hidden_layer_sizes=(3,3), solver='lbfgs', alpha=0.0001)
    )

    model_8 = make_pipeline(
        IPCA(),
        MLPClassifier(hidden_layer_sizes=(5,), solver='lbfgs', alpha=0.0001)
    )

    model = EnsembleVoteClassifier(clfs=[model_1, model_2, model_3, model_4, model_5, model_6, model_7, model_8], voting='soft', weights=[0,0,0,0,1,0,0,0])

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
