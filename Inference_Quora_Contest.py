from multiprocessing import Pool
import pandas as pd
import numpy as np
import joblib

def accuracy_calculation (predict, observe, testing_result_tracker):

    predict = predict.astype ('int64')
    observe = observe.astype ('int64')

    if len (testing_result_tracker):
        predict = np.append (testing_result_tracker[0], predict)
        observe = np.append (testing_result_tracker[1], observe)

    number_of_correct_predicting = np.sum (predict == observe)
    accuracy_percentage = number_of_correct_predicting / len (predict) * 100

    return np.array ([predict, observe]), round (accuracy_percentage, 4)

def LightGBM_process (quora_test):

    from LightGBM_lexical import LightGBM_Lexical
    
    LightGBM = joblib.load ('LightGBM_model.pkl')

    LightGBM_predictions = LightGBM.make_predictions (quora_test)
    return LightGBM_predictions

def SBERT_process (quora_test):

    from SBERT_embedding import SBERT_Embedding

    SBERT = SBERT_Embedding ()
    SBERT.load_model ('PEFT_model')

    SBERT_predictions = SBERT.predict (quora_test).cpu ().numpy ()
    return SBERT_predictions

def quora_testing_preparation ():

    quora_test = pd.read_csv ('quora-question-pairs/test.csv')
    quora_result = pd.read_csv ('quora-question-pairs/sample_submission.csv')

    try:
        rows_completed = joblib.load ('rows_completed.pkl')
        start_row_quora_test = rows_completed + 1 
    except:
        start_row_quora_test = 0

    number_of_rows_for_testing = 10
    end_row_quora_test = start_row_quora_test + number_of_rows_for_testing
    
    if end_row_quora_test >= quora_test.shape[0]:
        end_row_quora_test = quora_test.shape[0]
    new_rows_completed = end_row_quora_test - 1

    quora_test_rows_extracted = quora_test.iloc[start_row_quora_test : end_row_quora_test]    
    quora_result_rows_extracted = quora_result.iloc[start_row_quora_test : end_row_quora_test]

    return quora_test_rows_extracted, quora_result_rows_extracted, new_rows_completed

def get_testing_result_tracker ():

    try:
        testing_result_tracker = joblib.load ('testing_result_tracker.pkl')
    except:
        testing_result_tracker = np.array ([])

    return testing_result_tracker


SBERT_weight_selected = 0.45
LightGBM_weight_selected = 0.55
threshold_label_decision = joblib.load ('threshold_label_decision.pkl')


if __name__ == '__main__':

    quora_test, quora_result, new_rows_completed = quora_testing_preparation ()
    testing_result_tracker = get_testing_result_tracker ()

    with Pool () as pool:
        SBERT_predictions = pool.apply (SBERT_process, [quora_test])
    with Pool () as pool:
        LightGBM_predictions = pool.apply (LightGBM_process, [quora_test])
    
    SBERT_predictions_weighted = SBERT_predictions * SBERT_weight_selected
    LightGBM_predictions_weighted = LightGBM_predictions * LightGBM_weight_selected

    final_predictions = SBERT_predictions_weighted + LightGBM_predictions_weighted
    final_predictions = np.array ([1 if prediction <= threshold_label_decision else 0 for prediction in final_predictions])
    
    new_testing_result_tracker, accuracy_percentage = accuracy_calculation (final_predictions, quora_result['is_duplicate'].values, testing_result_tracker)

    print (f'After testing {new_rows_completed} samples, the models result in {accuracy_percentage}% accuracy')

    joblib.dump (accuracy_percentage, 'accuracy_until_rows_competed.pkl')
    joblib.dump (new_testing_result_tracker, 'testing_result_tracker.pkl')
    joblib.dump (new_rows_completed, 'rows_completed.pkl')