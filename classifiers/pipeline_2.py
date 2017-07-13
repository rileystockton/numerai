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
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, Normalizer
from mlxtend.classifier import EnsembleVoteClassifier
from sklearn import manifold
from operator import mul
import sys
sys.path.append('../functions')
from classifier_functions import *

def main():
    seed = 9000
    num_trees = 10
    script_name = os.path.basename(__file__).split('.')[0]
    # Set seed for reproducibility
    np.random.seed(seed)
    print("{} - Loading data...".format(script_name))
    # Load the data from the CSV files
    training_data = pd.read_csv('../data/numerai_training_data.csv')
    tournament_data = pd.read_csv('../data/numerai_tournament_data.csv')
    validation_data = tournament_data[tournament_data['data_type'] == 'validation']
    test_data = tournament_data[tournament_data['data_type'] == 'test']
    validation_data.is_copy = False
    test_data.is_copy = False
    ids = tournament_data['id']
    # Get ids and features
    features = [f for f in list(training_data) if "feature" in f]
    # features_plus_era = features + ['era']
    # training_data[features] = training_data[features_plus_era].groupby('era').transform(lambda x: (x - x.mean()))
    # validation_data[features] = validation_data[features_plus_era].groupby('era').transform(lambda x: (x - x.mean()))
    # test_data[features] = test_data[features_plus_era].groupby('era').transform(lambda x: (x - x.mean()))

    ### Feature Creation ###
    print("{} - Creating New Features...".format(script_name))

    # cor = training_data.corr()['target']
    # positive_correlators = [x for x in features if cor[x]>0.01]
    # print(positive_correlators)
    # negative_correlators = [x for x in features if cor[x]<-0.01]
    # print(negative_correlators)
    # for i in range(len(positive_correlators)-1):
    #   for cmb in combinations(positive_correlators, i+2):
    #       new_feature_training, new_feature_validation = 1, 1
    #       new_feature_name = "_x_".join(cmb)
    #       print(new_feature_name)
    #       for c in cmb:
    #           new_feature_training = new_feature_training * training_data[c]
    #           new_feature_validation = new_feature_validation * validation_data[c]
    #       training_data[new_feature_name] = new_feature_training
    #       validation_data[new_feature_name] = new_feature_validation
    # for i in range(len(negative_correlators)-1):
    #   for cmb in combinations(negative_correlators, i+2):
    #       new_feature_training, new_feature_validation = 1, 1
    #       new_feature_name = "_x_".join(cmb)
    #       print(new_feature_name)
    #       for c in cmb:
    #           new_feature_training = new_feature_training * training_data[c]
    #           new_feature_validation = new_feature_validation * validation_data[c]
    #       training_data[new_feature_name] = new_feature_training
    #       validation_data[new_feature_name] = new_feature_validation

    ### 2-combo multiplication ###
    # for cmb in combinations(features, 2):
    #     new_feature_training, new_feature_validation, new_feature_test = 1, 1, 1
    #     new_feature_name = "_x_".join(cmb)
    #     for c in cmb:
    #         new_feature_training = new_feature_training * training_data[c]
    #         new_feature_validation = new_feature_validation * validation_data[c]
    #         new_feature_test = new_feature_test * test_data[c]
    #     training_data[new_feature_name] = new_feature_training
    #     validation_data[new_feature_name] = new_feature_validation
    #     test_data[new_feature_name] = new_feature_test
    #
    # ### include all new features ###
    features = [f for f in list(training_data) if "feature" in f]
    # ### or use previously chosen set of features ###
    # # features = ['feature2', 'feature7', 'feature10', 'feature19', 'feature1_x_feature6', 'feature2_x_feature4', 'feature5_x_feature10', 'feature6_x_feature16', 'feature7_x_feature18', 'feature10_x_feature19', 'feature12_x_feature19', 'feature14_x_feature19', 'feature15_x_feature16', 'feature15_x_feature19', 'feature18_x_feature21']
    # ###
    #
    # ### reduce feature set ###
    # logreg = LogisticRegression()
    # rfecv = RFE(estimator=logreg, step=0.2, n_features_to_select=20)
    # features_train = training_data[features]
    # scaler = Normalizer()
    # print("{} - Scaling Feature Set...".format(script_name))
    # features_train = scaler.fit_transform(features_train)
    # print("{} - Reducing Feature Set...".format(script_name))
    # rfecv.fit(features_train, training_data['target'])
    # used_features = rfecv.get_support(indices=True).tolist()
    # reduced_features = [features[i] for i in used_features]
    # print("{} - {} Features Chosen".format(script_name, len(reduced_features)))
    # print(reduced_features)
    # features = reduced_features

    ### splitting train data into train and validation ###
    features_train, features_validation, targets_train, targets_validation = training_data[features], validation_data[features], training_data['target'], validation_data['target']

    labels = ['model_1', 'model_2', 'model_3', 'model']

    #.69216
    model_1 = make_pipeline(
        # PolynomialFeatures(degree=2, include_bias=False, interaction_only=False),
        StandardScaler(),
        PCA(svd_solver='randomized', iterated_power=10),
        # RFC(n_estimators=100, min_samples_split=10)
        GradientBoostingClassifier(n_estimators=25, min_samples_split=10)
    )

    # #.69249
    model_2 = make_pipeline(
        # PolynomialFeatures(degree=2, include_bias=False, interaction_only=False),
        Normalizer(),
        PCA(svd_solver='randomized', iterated_power=10),
        # RFC(n_estimators=100, min_samples_split=10)
        LogisticRegression(C=0.3)
    )
    #
    # #.69233
    model_3 = make_pipeline(
        # PolynomialFeatures(degree=2, include_bias=False, interaction_only=False),
        StandardScaler(),
        # PCA(svd_solver='randomized', iterated_power=10),
        RFC(n_estimators=25, max_depth=7)
    )

    model = EnsembleVoteClassifier(clfs=[model_1, model_2, model_3], voting='soft', weights=[3,1,1])

    # model = AdaBoostClassifier(base_estimator=model)
    print("{} - Training...".format(script_name))
    model.fit(features_train, targets_train)

    calculate_accuracy(script_name, model, features_validation, targets_validation)
    # predict probabilities for the validation set. returns list like [(prob of 0, prob of 1), (prob of 0, prob of 1), ...]
    prob_predictions_test = model.predict_proba(tournament_data[features])
    # get just probabilities of class = 1
    results = prob_predictions_test[:, 1]
    # convert to data frame
    results_df = pd.DataFrame(data={'probability':results})
    # join ids
    joined = pd.DataFrame(ids).join(results_df)
    print("{} - Writing...".format(script_name))
    # Save the predictions out to a CSV file
    joined.to_csv("../predictions/{}.csv".format(script_name), index=False)
    # Now you can upload these predictions on numer.ai

if __name__ == '__main__':
    main()
