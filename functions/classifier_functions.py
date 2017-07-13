import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import GridSearchCV as GS
from sklearn.ensemble import AdaBoostClassifier
from itertools import combinations
from sklearn import linear_model
from sklearn.feature_selection import RFE, RFECV
# from tsne import bh_sne
from sklearn.preprocessing import PolynomialFeatures
import time

def my_rfc(script_name, features, targets, n_estimators=[10], max_depth=[5], rfe_features=0):
    model = RFC()
    if rfe_features>0:
        parameters = {
           'estimator__n_estimators': n_estimators,
           'estimator__max_depth': max_depth
        }
        selector = RFECV(estimator=model, step=0.1)
        model = GS(selector, parameters)
    else:
        parameters = {
               'n_estimators': n_estimators,
               'max_depth': max_depth
        }
        model = GS(estimator=model, param_grid=parameters)
    print("{} - Training...".format(script_name))
    model.fit(features, targets)
    print("{} - Best Hyperparameters: {}".format(script_name,str(model.best_params_)))
    return model

def my_log_reg(script_name, feature_names, features, targets):
    model = linear_model.LogisticRegression()
    model = RFECV(estimator=model, step=0.1)
    print("{} - Training...".format(script_name))
    model.fit(features, targets)
    print_feature_ranks(script_name, feature_names, model)
    return model

def print_feature_ranks(script_name, feature_names, model):
    print(feature_names)
    print(model.ranking_)
    feature_ranks = pd.DataFrame({'feature' : feature_names, 'rank' : model.ranking_})
    print("{} - Feature Ranks:".format(script_name))
    print(feature_ranks)

def my_ada_rfc(script_name, features, targets, n_estimators=[10], max_depth=[5], rfe_features=0):
    model = RFC()
    parameters = {
       'base_estimator__n_estimators': n_estimators,
       'base_estimator__max_depth': max_depth
    }
    booster = AdaBoostClassifier(base_estimator=model)
    model = GS(booster, parameters)
    print("{} - Training...".format(script_name))
    model.fit(features, targets)
    print("{} - Best Hyperparameters: {}".format(script_name,str(model.best_params_)))
    return model

def calculate_accuracy(script_name, model, features_test, targets_test):
    if hasattr(model, 'predict_proba'):
        prob_predictions_class_test = model.predict(features_test)
        prob_predictions_test = model.predict_proba(features_test)
        accuracy = accuracy_score(targets_test, prob_predictions_class_test, normalize=True, sample_weight=None)
        logloss = log_loss(targets_test,prob_predictions_test)
        print("{} - Logloss: {}".format(script_name, logloss))
        print("{} - Accuracy: {}".format(script_name, accuracy))
    else:
        prob_predictions_test = model.predict(features_test)
        logloss = log_loss(targets_test,prob_predictions_test)
        print("{} - Logloss: {}".format(script_name, logloss))


# def my_tsne(features, features_train, features_test, targets_train, targets_test, tournament_data, perplexity, dimensions=2, polynomial=False):
#
#     X_train = features_train.values
#     y_train = targets_train.values
#
#     X_test = features_test.values
#     y_test = targets_test.values
#
#     X_tournament = tournament_data[features].values
#
#     X_all = np.concatenate([X_train, X_test, X_tournament], axis=0)
#
#     if polynomial:
#         poly = PolynomialFeatures(degree=2)
#         X_all = poly.fit_transform(X_all)
#
#     print('Running TSNE (perplexity: {}, dimensions: {}, polynomial: {})...'.format(perplexity, dimensions, polynomial))
#     start_time = time.time()
#     tsne_all = bh_sne(X_all, d=dimensions, perplexity=float(perplexity))
#     print('TSNE: {}s'.format(time.time() - start_time))
#
#     tsne_train = tsne_all[:X_train.shape[0]]
#     assert(len(tsne_train) == len(X_train))
#     features_train['tsne_feature_1'] = [x[0] for x in tsne_train]
#     features_train['tsne_feature_2'] = [x[1] for x in tsne_train]
#
#     tsne_test = tsne_all[X_train.shape[0]:X_train.shape[0]+X_test.shape[0]]
#     assert(len(tsne_test) == len(X_test))
#     features_test['tsne_feature_1'] = [x[0] for x in tsne_test]
#     features_test['tsne_feature_2'] = [x[1] for x in tsne_test]
#
#     tsne_tournament = tsne_all[X_train.shape[0]+X_test.shape[0]:X_train.shape[0]+X_test.shape[0]+X_tournament.shape[0]]
#     assert(len(tsne_tournament) == len(X_tournament))
#     tournament_data['tsne_feature_1'] = [x[0] for x in tsne_tournament]
#     tournament_data['tsne_feature_2'] = [x[1] for x in tsne_tournament]
#
#     return features_train, features_test, tournament_data
