from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem import WordNetLemmatizer
from rank_bm25 import BM25Okapi
import lightgbm as lgb
import pandas as pd
import numpy as np
import unicodedata
import contractions
import Levenshtein
import re
import joblib

class LightGBM_Lexical:

    def __init__ (self, file):
        
        self.model = lgb.LGBMClassifier (boosting_type = 'goss',
                                        n_estimators = 1000,
                                        subsample = 0.8,
                                        n_jobs = -1)
        
        self.lemmatizer = WordNetLemmatizer ()

        self.Bag_of_Words = CountVectorizer (stop_words = 'english', max_features = 50000)
        self.Tf_Idf = TfidfVectorizer (stop_words = 'english', max_features = 50000)
        self.BM_25 = None

        self.pre_train_with_corpus (file)

    def pre_train_with_corpus (self, file):

        corpus = pd.concat ([file['question1'], file['question2']]).unique ()
        self.Bag_of_Words.fit (corpus)
        self.Tf_Idf.fit (corpus)

        corpus = [self.tokenize (question) for question in corpus]
        self.BM_25 = BM25Okapi (corpus)
    
    def tokenize (self, text):
        
        text = text.lower()
        text = unicodedata.normalize ('NFKD', text)
        text = contractions.fix (text)

        pattern = r"\w+(?:['-_]\w+)*|[^\w\s]+"
        text = re.findall (pattern, text)
        text = [self.lemmatizer.lemmatize (token) for token in text]

        return text

    def len_str_ratio (self, q1, q2):

        if len (q2) < len (q1):
            q1, q2 = q2, q1

        return len (q1) / (len (q2) + 1e-9)

    def distinct_token_and_word_and_number_and_other (self, q1, q2):

        token_1, token_2 = {}, {}
        word_1, word_2 = {}, {}
        num_1, num_2 = {}, {}
        other_1, other_2 = {}, {}

        for word in q1:
            token_1[word] = 1
            if word.isalpha ():
                word_1[word] = 1
            if word.isnumeric ():
                num_1[word] = 1
            if not word.isalnum ():
                other_1[word] = 1

        for word in q2:
            token_2[word] = 1
            if word.isalpha ():
                word_2[word] = 1
            if word.isnumeric ():
                num_2[word] = 1
            if not word.isalnum ():
                other_2[word] = 1
        
        token_count = self.len_str_ratio (token_1, token_2)
        word_count = self.len_str_ratio (word_1, word_2)
        num_count = self.len_str_ratio (num_1, num_2)
        other_count = self.len_str_ratio (other_1, other_2)

        return token_count, word_count, num_count, other_count
    
    def word_overlap_and_Jaccard_similarity (self, tokenized_q1, tokenized_q2):

        set_q1 = set (tokenized_q1)
        set_q2 = set (tokenized_q2)

        intersection = set_q1.intersection (set_q2)
        union = set_q1.union (set_q2)

        overlap = 2 * len (intersection) / (len (tokenized_q1) + len (tokenized_q2))
        Jaccard_similarity = len (intersection) / len (union)

        return overlap, Jaccard_similarity

    def token_fuzzy_ratio (self, q1, q2):

        Levenshtein_dist = Levenshtein.distance (q1, q2)
        nominator = len (q1) + len (q2) - Levenshtein_dist
        denominator = len (q1) + len (q2) + 1e-9

        return nominator / denominator * 100

    def token_sort_ratio (self, q1, q2):

        q1.sort ()
        q2.sort ()

        Levenshtein_dist = Levenshtein.distance (q1, q2)
        nominator = len (q1) + len (q2) - Levenshtein_dist
        denominator = len (q1) + len (q2) + 1e-9

        return nominator / denominator * 100

    def fuzzy_partial_ratio (self, q1, q2):

        if len (q2) < len (q1):
            q1, q2 = q2, q1
        
        max_partial_ratio = 0
        for i in range (len (q2) - len (q1) + 1):
            string = q2[i : i + len (q1)]

            Levenshtein_dist = Levenshtein.distance (q1, string)
            nominator = len (q1) + len (string) - Levenshtein_dist
            denominator = len (q1) + len (string) + 1e-9

            max_partial_ratio = max (max_partial_ratio, nominator / denominator * 100)
        
        return max_partial_ratio

    def bag_of_words (self, q1, q2):
        
        q1_bow = self.Bag_of_Words.transform ([q1])
        q2_bow = self.Bag_of_Words.transform ([q2])

        bow_cosine_similarity = cosine_similarity (q1_bow, q2_bow)[0][0]
        return bow_cosine_similarity

    def tf_idf (self, q1, q2):

        q1_tf_idf = self.Tf_Idf.transform ([q1])
        q2_tf_idf = self.Tf_Idf.transform ([q2])

        tf_idf_cosine_similarity = cosine_similarity (q1_tf_idf, q2_tf_idf)[0][0]
        return tf_idf_cosine_similarity


    def bm_25 (self, q1, q2):

        q1_bm_25 = self.BM_25.get_scores (q1).reshape (1, -1)
        q2_bm_25 = self.BM_25.get_scores (q2).reshape (1, -1)

        bm_25_cosine_similarity = cosine_similarity (q1_bm_25, q2_bm_25)[0][0]
        return bm_25_cosine_similarity


    def input_creation (self, file):
        
        X = []
        for row in file.itertuples ():
            raw_q1 = row.question1
            raw_q2 = row.question2
            
            tokenized_q1 = self.tokenize (raw_q1)
            tokenized_q2 = self.tokenize (raw_q2)

            feature_1 = self.len_str_ratio (raw_q1, raw_q2)
            feature_2, feature_3, feature_4, feature_5 = self.distinct_token_and_word_and_number_and_other (tokenized_q1, tokenized_q2)
            feature_6, feature_7 = self.word_overlap_and_Jaccard_similarity (tokenized_q1, tokenized_q2)
            feature_8 = self.token_fuzzy_ratio (tokenized_q1, tokenized_q2)
            feature_9 = self.token_sort_ratio (tokenized_q1, tokenized_q2)
            feature_10 = self.fuzzy_partial_ratio (tokenized_q1, tokenized_q2)
            feature_11 = self.bag_of_words (raw_q1, raw_q2)
            feature_12 = self.tf_idf (raw_q1, raw_q2)
            feature_13 = self.bm_25 (tokenized_q1, tokenized_q2)

            X.append ([feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, 
                        feature_7, feature_8, feature_9, feature_10, feature_11, feature_12,
                        feature_13])
        
        return X

    def train (self, file):

        X_train = np.array (self.input_creation (file))
        Y_train = file['is_duplicate'].values
        self.model.fit (X_train, Y_train)

    def make_predictions (self, file):

        X_test = np.array (self.input_creation (file))
        predictions = self.model.predict_proba (X_test)

        probability_positive_class = predictions[:, 1]
        return probability_positive_class


if __name__ == '__main__':

    dataset = pd.read_csv ('quora-question-pairs/train.csv')
    train_test_split = joblib.load ('train_test_split.pkl')
    train_indices_list = train_test_split['train_dataset']
    test_indices_list = train_test_split['test_dataset']

    dataset['question1'] = dataset['question1'].fillna ('')
    dataset['question2'] = dataset['question2'].fillna ('')
    LightGBM_Lexical_score = []

    for index in range (len (train_indices_list)):

        train_index = train_indices_list[index]
        test_index = test_indices_list[index]

        train_dataset = dataset.iloc[train_index]
        test_dataset = dataset.iloc[test_index]

        model = LightGBM_Lexical (train_dataset)
        model.train (train_dataset)
        probability_positive_class = model.make_predictions (test_dataset)

        LightGBM_Lexical_score.append (probability_positive_class)
    
    LightGBM_Lexical_score = np.array (LightGBM_Lexical_score)
    joblib.dump (LightGBM_Lexical_score, 'LightGBM_Lexical_Score.pkl')