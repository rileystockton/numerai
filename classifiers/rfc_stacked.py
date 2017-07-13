import os
import pandas as pd
import numpy as np
import sys
sys.path.append('/Users/rileystockton/Desktop/numerai/functions')
from classifier_functions import *
from stacked_generalizer import *
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.ensemble import ExtraTreesClassifier as ETC
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import GridSearchCV as GS
from sklearn.ensemble import AdaBoostClassifier
from itertools import combinations
from sklearn import linear_model
from sklearn.feature_selection import RFE


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
    features_train, features_test, targets_train, targets_test = train_test_split(training_data[features], training_data['target'], test_size=0.3, random_state=0)
    features_train, features_test, targets_train, targets_test =  features_train.values, features_test.values, targets_train.values, targets_test.values
    print(features_train)# define base models
    initial_model = my_rfc(script_name, features_train, targets_train, n_estimators=[25], max_depth=[5], rfe_features=10)
    base_models = [initial_model]
    # define blending model
    blending_model = linear_model.LogisticRegression()
    # initialize multi-stage model
    model = StackedGeneralizer(base_models, blending_model,
    	                    n_folds=5, verbose=True)
    print("{} - Training...".format(script_name))
    model.fit(features_train, targets_train)
    # calculate accuracy and logloss on test set
    calculate_accuracy(script_name, model, features_test, targets_test)
    # predict probabilities for the tournament set. returns list like [(prob of 0, prob of 1), (prob of 0, prob of 1), ...]
    prob_predictions_tournament = model.predict(tournament_data[features])
    # get just probabilities of class = 1
    results = prob_predictions_tournament[:, 1]
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
