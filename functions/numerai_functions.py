import pandas as pd
from sklearn.model_selection import train_test_split

def get_numerai_data(path):
    training_data = pd.read_csv(path + 'train_data.csv')
    tournament_data = pd.read_csv(path + 'numerai_tournament_data.csv')
    test_data = pd.read_csv(path + 'numerai_tournament_data.csv')
    validation_data = pd.read_csv(path + 'valid_data.csv')
    training_data = training_data.reset_index()
    test_data = test_data.reset_index()
    validation_data = validation_data.reset_index()
    tournament_ids = tournament_data['id']
    targets_train = training_data['target']
    targets_validation = validation_data['target']
    return training_data, validation_data, test_data, tournament_data, tournament_ids, targets_train, targets_validation

def get_numerai_data_old(path):
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
