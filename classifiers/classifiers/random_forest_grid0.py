import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import GridSearchCV as GS

def main():
    script_name = os.path.basename(__file__).split('.')[0]
    # Set seed for reproducibility
    np.random.seed(9000)
    print("{} - Loading data...".format(script_name))
    # Load the data from the CSV files
    training_data = pd.read_csv('../data/numerai_training_data.csv')
    tournament_data = pd.read_csv('../data/numerai_tournament_data.csv')
    # Get ids and features
    ids = tournament_data['id']
    features = [f for f in list(training_data) if "feature" in f]
    # splitting my arrays in ratio of 30:70 percent
    features_train, features_test, labels_train, labels_test = train_test_split(training_data[features], training_data['target'], test_size=0.3, random_state=0)
    # setting the range of values for our parameters
    parameters = {
           'n_estimators': [20, 25],
           'random_state': [0],
           'max_features': [2],
           'min_samples_leaf': [150,200,250]
    }
    # Define classifier
    model = RFC()
    grid = GS(estimator=model, param_grid=parameters)
    # Training classifier
    print("{} - Training...".format(script_name))
    grid.fit(features_train, labels_train)
    # Calculate the logloss of the model
    prob_predictions_class_test = grid.predict(features_test)
    prob_predictions_test = grid.predict_proba(features_test)
    logloss = log_loss(labels_test,prob_predictions_test)
    accuracy = accuracy_score(labels_test, prob_predictions_class_test, normalize=True, sample_weight=None)
    print 'accuracy', accuracy
    print 'logloss', logloss
    # predict probabilities for the tournament set. returns list like [(prob of 0, prob of 1), (prob of 0, prob of 1), ...]
    prob_predictions_tournament = grid.predict_proba(tournament_data[features])
    # get just probabilities of 1
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
