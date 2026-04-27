import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import joblib


def ROC_point_collection (predict, observe):

    roc = []
    output_threshold = np.unique (np.array (predict))

    for threshold in output_threshold:
        confu_matrix = np.zeros ((2, 2), dtype = 'int64')
        temp_predict = np.array ([1 if score >= threshold else 0 for score in predict])

        for i in range (len (temp_predict)):
            prediction = np.int64 (temp_predict[i])
            obbservation = np.int64 (observe[i])
            confu_matrix[prediction][obbservation] += 1

        TP = confu_matrix[1][1]
        FP = confu_matrix[1][0]
        TN = confu_matrix[0][0]
        FN = confu_matrix[0][1]

        FP_rate = FP / (FP + TN)
        TP_rate = TP / (TP + FN)

        roc.append ((FP_rate, TP_rate))
    
    roc.sort (key = lambda x: x[0])
    return roc

def AUC (roc):

    area_under_curve = 0
    for index in range (1, len (roc)):

        point_current = roc[index]
        point_last = roc[index - 1]

        x_axis_length = point_current[0] - point_last[0]
        y_axis_length = (point_current[1] + point_last[1]) / 2
        area_under_curve += x_axis_length * y_axis_length
    
    return '{:.2f}'.format (area_under_curve)

def plotting (ROC_info):

    fig, ((graph1, graph2, graph3), (graph4, graph5, graph6), (graph7, graph8, graph9)) = plt.subplots (3, 3, figsize = (16, 10))
    graphs = [graph1, graph2, graph3, graph4, graph5, graph6, graph7, graph8, graph9]

    for index, graph in enumerate (graphs):

        ROC_datapoint_for_graph_i = ROC_info[index][0]
        AUC_for_graph_i = ROC_info[index][1]
        SBERT_weight_for_graph_i, LightGBM_weight_for_graph_i = ROC_info[index][2]
        
        x_axis = []
        y_axis = []
        for point in ROC_datapoint_for_graph_i:
            x_axis.append (point[0])
            y_axis.append (point[1])

        graph.plot (x_axis, y_axis, label = f'AUC = {AUC_for_graph_i}', color = 'blue')
        graph.plot ([0, 1], [0, 1], 'r--')
        graph.set_title (f'SBERT weight: {SBERT_weight_for_graph_i} and LightGBM weight: {LightGBM_weight_for_graph_i}')
        graph.set_ylabel ('True Positive Rate')
        graph.set_xlabel ('False Positive Rate')
        graph.legend (loc = 'best')
    
    plt.tight_layout ()
    plt.savefig ('ROC_AUC_Grid_Search.png')


dataset = pd.read_csv ('quora-question-pairs/train.csv')
train_test_split = joblib.load ('train_test_split.pkl')
test_indices_list = train_test_split['test_dataset']

SBERT_embedding_score = joblib.load ('SBERT_Embedding_Score.pkl')
LightGBM_lexical_score = joblib.load ('LightGBM_Lexical_Score.pkl')

grid_search = [[0.05, 0.95],
                [0.1, 0.9],
                [0.15, 0.85],
                [0.2, 0.8],
                [0.25, 0.75],
                [0.3, 0.7],
                [0.35, 0.65],
                [0.4, 0.6],
                [0.45, 0.55]]


if __name__ == '__main__':

    ROC_graph = []
    for weight in grid_search:
        SBERT_weight = weight[0]
        LightGBM_weight = weight[1]

        prediction = []
        observation = []

        for index in range (len (test_indices_list)):

            test_index = test_indices_list[index]
            test_dataset = dataset.iloc[test_index]
            test_dataset_output = list (test_dataset['is_duplicate'].values)
            observation.extend (test_dataset_output)

            SBERT_score = SBERT_embedding_score[index] * SBERT_weight
            LightGBM_score = LightGBM_lexical_score[index] * LightGBM_weight
            positive_output = SBERT_score + LightGBM_score
            prediction.extend (list (positive_output))

        ROC_graph_i = ROC_point_collection (prediction, observation)
        AUC_graph_i = AUC (ROC_graph_i)
        ROC_graph.append ((ROC_graph_i, AUC_graph_i, (SBERT_weight, LightGBM_weight)))

    plotting (ROC_graph)