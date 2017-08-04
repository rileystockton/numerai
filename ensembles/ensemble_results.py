import sys
import pandas as pd
from sklearn.metrics import log_loss
import itertools

def main():
    final = sys.argv[1]
    final_string = ""
    if final == "1":
        final_string = "_final"
    result_file = '../ensembles/ensemble_results.csv'
    classifiers = {
        'gbc_4-21': 1,
        # 'gbc_5-17': 0,
        # 'gbc_10-19': 0,
        'gbc_16-18': 0,
        'rfc_10_5': 0,
        # 'rfc_10_4': 0,
        # 'rfc_5_5': 0,
        # 'rfc_3_5': 0,
        # 'mlp_3_0.001': 0,
        # 'mlp_3_1e-05': 0,
        'mlp_5_0.001': 1,
        # 'mlp_5_1e-05': 0,
        'linear_2': 0,
        'logistic': 0
    }

    ### initialize data frame ###
    column_list = list(classifiers)
    column_list.append('result')
    df = pd.DataFrame(columns=column_list)

    for combo in itertools.product([0,1], repeat=len(classifiers)):
        try:
            classifiers.update(dict(zip(classifiers.keys(), combo)))
            i = 0
            sum_weight = 0
            for clf, weight in classifiers.iteritems():
                df_test = pd.read_csv('../predictions/' + clf + '{}.csv'.format(final_string))
                df_test = df_test.rename(columns = {'probability' : clf})
                df_validation = pd.read_csv('../predictions/' + clf + '_validation{}.csv'.format(final_string))
                logloss = log_loss(df_validation['target'],df_validation['probability'])
                df_validation = df_validation.rename(columns = {'probability' : clf})
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
            logloss = log_loss(full_df_validation['target'],full_df_validation['probability'])
            new_row = list(combo)
            new_row.append(logloss)
            print new_row
            new_row = pd.DataFrame([new_row], columns=column_list)
            df = df.append(new_row)
        except KeyboardInterrupt:
            raise
        except:
            'oops!'
    df = df.sort_values(['result'], ascending=[1])
    df.to_csv(result_file, index=False)

if __name__ == '__main__':
    main()
