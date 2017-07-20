import sys
import pandas as pd
from sklearn.metrics import log_loss

def main():
    final = sys.argv[1]
    final_string = ""
    if final == "1":
        final_string = "_final"
    ### define classiers to and their respective weights ###
    classifiers = ['gbc_4-21', 'gbc_5-17', 'gbc_10-19', 'gbc_16-18', 'rfc', 'mlp', 'logistic']
    weights = [1, 1, 1, 1, 1, 1, 1]

    ### create ensemble projections ###
    for i in range(len(classifiers)):
        clf = classifiers[i]
        df_test = pd.read_csv('../predictions/' + clf + '{}.csv'.format(final_string))
        df_test = df_test.rename(columns = {'probability' : clf})
        df_validation = pd.read_csv('../predictions/' + clf + '_validation{}.csv'.format(final_string))
        logloss = log_loss(df_validation['target'],df_validation['probability'])
        df_validation = df_validation.rename(columns = {'probability' : clf})
        print "{} - logloss: {}".format(clf, logloss)
        if i == 0:
            full_df_test = df_test
            full_df_test['probability'] = full_df_test[classifiers[i]] * weights[i]
            full_df_validation = df_validation
            full_df_validation['probability'] = full_df_validation[classifiers[i]] * weights[i]
        else:
            full_df_test = pd.merge(full_df_test, df_test, how='left', on='id')
            full_df_test['probability'] = full_df_test['probability'] + full_df_test[classifiers[i]] * weights[i]
            full_df_validation = pd.merge(full_df_validation, pd.DataFrame(df_validation[clf]), left_index=True, right_index=True)
            full_df_validation['probability'] = full_df_validation['probability'] + full_df_validation[classifiers[i]] * weights[i]

    full_df_test['probability'] = full_df_test['probability'] / sum(weights)
    full_df_validation['probability'] = full_df_validation['probability'] / sum(weights)
    print(full_df_test.head(5))
    print(full_df_validation.head(5))
    logloss = log_loss(full_df_validation['target'],full_df_validation['probability'])
    print "Ensemble - logloss: {}".format(logloss)

    final_probabilities = full_df_test[['id','probability']]
    print("Ensemble - Writing...")
    final_probabilities.to_csv("../predictions/hal9001.csv", index=False)

if __name__ == '__main__':
    main()
