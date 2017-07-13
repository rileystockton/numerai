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
from sklearn.feature_selection import RFE
import sys
sys.path.append('/Users/rileystockton/Desktop/numerai/functions')
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
    # Get ids and features
    ids = tournament_data['id']
    features = [f for f in list(training_data) if "feature" in f]
    for cmb in combinations(features, 2):
        if int(cmb[0].replace('feature','')) % 2 == 0 and int(cmb[1].replace('feature','')) % 2 == 1:
            new_feature_name = '{}_over_{}'.format(cmb[0],cmb[1])
            new_feature_training = training_data[cmb[0]] / (training_data[cmb[1]] + 0.01)
            new_feature_tournament = tournament_data[cmb[0]] / (tournament_data[cmb[1]] + 0.01)
            training_data[new_feature_name] = new_feature_training
            tournament_data[new_feature_name] = new_feature_tournament
    for feature in features:
        new_feature_name = 'log_{}'.format(feature)
        new_feature_training = np.log(training_data[feature]+0.01)
        new_feature_tournament = np.log(tournament_data[feature]+0.01)
        training_data[new_feature_name] = new_feature_training
        tournament_data[new_feature_name] = new_feature_tournament
    features = [f for f in list(training_data) if "feature" in f]
    # splitting my arrays in ratio of 30:70 percent
    features_train, features_test, targets_train, targets_test = train_test_split(training_data[features], training_data['target'], test_size=0.3, random_state=0)
    # setting the range of values for our parameters
    model = my_rfc(script_name, features_train, targets_train, n_estimators=[25], max_depth=[5], rfe_features=10)
    # calculate accuracy and logloss on test set
    calculate_accuracy(script_name, model, features_test, targets_test)
    # predict probabilities for the tournament set. returns list like [(prob of 0, prob of 1), (prob of 0, prob of 1), ...]
    prob_predictions_tournament = model.predict_proba(tournament_data[features])
    # get just probabilities of class = 1
    results = prob_predictions_tournament[:, 1]
    results = [.5 if (p > .45 and p < .55) else p for p in results]
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
