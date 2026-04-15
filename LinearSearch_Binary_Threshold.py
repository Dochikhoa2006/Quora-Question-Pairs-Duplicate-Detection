import pandas as pd
import numpy as np
import joblib

def accuracy_calculation (predict, observe):

    predict = predict.astype ('int64')
    observe = observe.astype ('int64')

    number_of_correct_predicting = 0
    for index in range (len (predict)):
        if predict[index] == observe[index]:
            number_of_correct_predicting += 1
    
    accuracy_percentage = number_of_correct_predicting / len (predict)
    return accuracy_percentage


dataset = pd.read_csv ('/Users/chikhoado/Desktop/PROJECTS/Quora Question/quora-question-pairs/train.csv')
train_test_split = joblib.load ('train_test_split.pkl')
test_indices_list = train_test_split['test_dataset']

SBERT_embedding_score = joblib.load ('SBERT_Embedding_Score.pkl')
LightGBM_lexical_score = joblib.load ('LightGBM_Lexical_Score.pkl')

SBERT_weight_selected = 0.45
LightGBM_weight_selected = 0.55

prediction = []
observation = []

for index in range (len (test_indices_list)):

    test_index = test_indices_list[index]
    test_dataset = dataset.iloc[test_index]
    test_dataset_output = test_dataset['is_duplicate'].values
    observation.append (test_dataset_output)

    SBERT_score = SBERT_embedding_score[index] * SBERT_weight_selected
    LightGBM_score = LightGBM_lexical_score[index] * LightGBM_weight_selected
    positive_output = SBERT_score + LightGBM_score
    prediction.append (positive_output)

best_binary_classification_threshold = 0
best_accuracy_percentage = 0

for index in np.arange (0.01, 1, 0.01):
    accuracy_percentage = 0

    for idx in range (len (prediction)):

        predict = prediction[idx]
        predict_with_threshold = np.array ([1 if score >= index else 0 for score in predict])
        observe = observation[idx]
        accuracy_percentage += accuracy_calculation (predict_with_threshold, observe)
    
    accuracy_percentage = accuracy_percentage / len (prediction)

    if accuracy_percentage > best_accuracy_percentage:
        best_accuracy_percentage = accuracy_percentage
        best_binary_classification_threshold = index

best_binary_classification_threshold = round (best_binary_classification_threshold, 2)
joblib.dump (best_binary_classification_threshold, 'threshold_label_decision.pkl')





# cd '/Users/chikhoado/Desktop/PROJECTS/Quora Question'
# /opt/homebrew/bin/python3.12 -m venv .venv
# source .venv/bin/activate
# pip install pandas numpy
# python '/Users/chikhoado/Desktop/PROJECTS/Quora Question/LinearSearch_Binary_Threshold.py'