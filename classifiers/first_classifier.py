#!/usr/bin/env python

"""
Example classifier on Numerai data using a logistic regression classifier.
To get started, install the required packages: pip install pandas, numpy, sklearn
"""

import pandas as pd
import numpy as np
import os
from sklearn import metrics, preprocessing, linear_model


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
    X = training_data[features]
    Y = training_data["target"]
    x_prediction = prediction_data[features]
    ids = prediction_data["id"]

    # This is your model that will learn to predict
    model = linear_model.LogisticRegression(n_jobs=-1)

    print("{} - Training...".format(script_name))
    # Your model is trained on the training_data
    model.fit(X, Y)

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
