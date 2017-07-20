import pandas as pd
from sklearn.model_selection import train_test_split

def get_numerai_data(path):
    training_data = pd.read_csv(path + 'numerai_training_data.csv')
    tournament_data = pd.read_csv(path + 'numerai_tournament_data.csv')
    validation_data = tournament_data[tournament_data['data_type'] == 'validation']
    test_data = tournament_data[tournament_data['data_type'] == 'test']
    validation_data.is_copy = False
    test_data.is_copy = False
    tournament_ids = tournament_data['id']
    targets_train = training_data['target']
    targets_validation = validation_data['target']
    return training_data, validation_data, test_data, tournament_data, tournament_ids, targets_train, targets_validation
