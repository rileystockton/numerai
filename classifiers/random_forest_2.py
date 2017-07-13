#!/usr/bin/env python

"""
Example classifier on Numerai data using a logistic regression classifier.
To get started, install the required packages: pip install pandas, numpy, sklearn
"""

import pandas as pd
import numpy as np
import os
from sklearn import metrics, preprocessing, linear_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.metrics import accuracy_score, log_loss
from itertools import combinations
from scipy.stats.stats import pearsonr
from sklearn.linear_model import RandomizedLasso
import operator


def main():
    script_name = os.path.basename(__file__).split('.')[0]
    # Set seed for reproducibility
    np.random.seed(9000)

    print("{} - Loading data...".format(script_name))
    # Load the data from the CSV files
    training_data = pd.read_csv('../data/numerai_training_data.csv', header=0)
    prediction_data = pd.read_csv('../data/numerai_tournament_data.csv', header=0)

    # Transform the loaded CSV data into numpy arrays
    features = [f for f in list(training_data) if "feature" in f]
    target_list = training_data['target']
    features = [f for f in list(training_data) if "feature" in f]
    # features = ['feature5','feature7','feature6','feature9','feature10','feature16','feature19']
    cmb_num=0
    for cmb in combinations(features, 2):
        cmb_num = cmb_num+1
        new_feature_name = '{}_over_{}'.format(cmb[0],cmb[1])
        new_feature_training = training_data[cmb[0]] / (training_data[cmb[1]] + 0.01)
        new_feature_prediction = prediction_data[cmb[0]] / (prediction_data[cmb[1]] + 0.01)
        training_data[new_feature_name] = new_feature_training
        prediction_data[new_feature_name] = new_feature_prediction
    features = [f for f in list(training_data) if "feature" in f]
    print(len(features))
    pearson_scores = {}
    for feature in features:
        pearson_scores[feature] = pearsonr(training_data[feature],training_data['target'])[0]
    bottom_x = sorted(pearson_scores.items(), key=operator.itemgetter(1))[:50]
    top_x = sorted(pearson_scores.items(), key=operator.itemgetter(1))[-50:]
    bottom_features = map(list, zip(*bottom_x))[0]
    top_features = map(list, zip(*top_x))[0]
    features = bottom_features + top_features
    X = training_data[features]
    print(X.head(5))
    Y = training_data["target"]
    x_prediction = prediction_data[features]
    ids = prediction_data["id"]

    # This is your model that will learn to predict
    model = linear_model.LogisticRegression()
    # model = RFE(model,1)

    print("{} - Training...".format(script_name))
    # Your model is trained on the training_data
    model.fit(X, Y)
    # summarize the selection of the attributes
    print(model.ranking_)

    print("{} - Predicting...".format(script_name))
    # Your trained model is now used to make predictions on the numerai_tournament_data
    # The model returns two columns: [probability of 0, probability of 1]
    # We are just interested in the probability that the target is 1.
    y_prediction = model.predict_proba(x_prediction)

    results = y_prediction[:, 1]
    results_df = pd.DataFrame(data={'probability':results})
    joined = pd.DataFrame(ids).join(results_df)

    print("{} - Writing...".format(script_name))
    # Save the predictions out to a CSV file
    joined.to_csv("../predictions/{}.csv".format(script_name), index=False)
    # Now you can upload these predictions on numer.ai

if __name__ == '__main__':
    main()
