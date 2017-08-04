import sys
import pandas as pd
from sklearn.metrics import log_loss

def main():
    final = sys.argv[1]
    final_string = ""
    if final == "1":
        final_string = "_final"
    classifiers = {
        'gbc_4-21': 1,
        'gbc_5-17': 1,
        'gbc_10-19': 1,
        'gbc_16-18': 10,
        'rfc_10_5': 0,
        'rfc_10_4': 0,
        'rfc_5_5': 0,
        'rfc_3_5': 0,
        'mlp_3_0.001': 0,
        'mlp_3_1e-05': 0,
        'mlp_5_0.001': 0,
        'mlp_5_1e-05': 0,
        'linear_2': 1,
        'logistic': 1
    }

    ### create ensemble projections ###
    i = 0
    sum_weight = 0
    for clf, weight in classifiers.iteritems():
        df_test = pd.read_csv('../predictions/' + clf + '{}.csv'.format(final_string))
        df_test = df_test.rename(columns = {'probability' : clf})
        df_validation = pd.read_csv('../predictions/' + clf + '_validation{}.csv'.format(final_string))
        logloss = log_loss(df_validation['target'],df_validation['probability'])
        df_validation = df_validation.rename(columns = {'probability' : clf})
        print "{}: {}".format(clf, logloss)
        if i == 0:
            full_df_test = df_test
            full_df_test['probability'] = full_df_test[clf] * weight
            full_df_validation = df_validation
            full_df_validation['probability'] = full_df_validation[clf] * weight
        else:
            full_df_test = pd.merge(full_df_test, df_test, how='left', on='id')
            full_df_test['probability'] = full_df_test['probability'] + full_df_test[clf] * weight
            full_df_validation = pd.merge(full_df_validation, pd.DataFrame(df_validation[clf]), left_index=True, right_index=True)
            full_df_validation['probability'] = full_df_validation['probability'] + full_df_validation[clf] * weight
        i = i + 1
        sum_weight = sum_weight + weight

    full_df_test['probability'] = full_df_test['probability'] / sum_weight
    full_df_validation['probability'] = full_df_validation['probability'] / sum_weight
    # print(full_df_test.head(5))
    # print(full_df_validation.head(5))
    logloss = log_loss(full_df_validation['target'],full_df_validation['probability'])
    print "Ensemble - logloss: {}".format(logloss)

    final_probabilities = full_df_test[['id','probability']]
    print("Ensemble - Writing...")
    final_probabilities.to_csv("../predictions/hal9002{}.csv".format(final_string), index=False)

if __name__ == '__main__':
    main()
