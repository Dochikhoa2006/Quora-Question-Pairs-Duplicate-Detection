from sentence_transformers import SentenceTransformer, util, InputExample
from sentence_transformers.sentence_transformer import losses
from torch.utils.data import DataLoader
from peft import LoraConfig, get_peft_model, PeftModel
import pandas as pd
import numpy as np
import random
import joblib 


class SBERT_Embedding:

    def __init__(self):
        
        self.SBERT = SentenceTransformer ('all-mpnet-base-v2')
        self.LoRA_configuration = LoraConfig (r = 32,
                                target_modules = ['query', 'key', 'value', 'dense', 'intermediate.dense', 'output.dense'],
                                lora_dropout = 0.07,
                                bias = 'none')
        self.LoRA = get_peft_model (self.SBERT[0].auto_model, self.LoRA_configuration)
        self.SBERT[0].auto_model = self.LoRA

        self.MNRL = losses.MultipleNegativesRankingLoss (model = self.SBERT, similarity_fct = util.dot_score)
        self.CSL = losses.CosineSimilarityLoss (model = self.SBERT)
        self.TL = losses.TripletLoss (model = self.SBERT, 
                                distance_metric = losses.TripletDistanceMetric.EUCLIDEAN)
        self.LL = losses.SoftmaxLoss (model = self.SBERT, 
                                embedding_dimension = self.SBERT.get_embedding_dimension (),
                                num_labels = 2)
    
    def save_model (self):

        self.LoRA.save_pretrained ('PEFT_model')
            
    def load_model (self, LoRA_path):

        self.SBERT[0].auto_model = PeftModel.from_pretrained (self.SBERT[0].auto_model, LoRA_path)
    
    def predict (self, file):

        self.SBERT.eval ()
        batch_size = 32

        question1 = file['question1'].tolist () 
        question2 = file['question2'].tolist ()

        question1_embedding = self.SBERT.encode (question1, batch_size = batch_size, show_progress_bar = True)
        question2_embedding = self.SBERT.encode (question2, batch_size = batch_size, show_progress_bar = True)

        cosine_similarity = util.pairwise_cos_sim (question1_embedding, question2_embedding)
        return cosine_similarity

    def fine_tune (self, file):

        question_pair_package_true, merge_package_label, triplet_package = self.input_creation (file)
        loader_mnrl, loader_csl, loader_tl, loader_ll = self.data_loader (question_pair_package_true, merge_package_label, triplet_package)

        self.SBERT.fit (train_objectives = [(loader_mnrl, self.MNRL),
                                            (loader_csl, self.CSL),
                                            (loader_tl, self.TL),
                                            (loader_ll, self.LL)],
                        epochs = 1,
                        warmup_steps = 70,
                        save_best_model = True,
                        show_progress_bar = True)
        
    def input_creation (self, file):

        file['question1'] = file['question1'].fillna ('')
        file['question2'] = file['question2'].fillna ('')

        file_true = file[file['is_duplicate'] == 1]
        file_false = file[file['is_duplicate'] == 0]

        question_pair_package_true = []
        for row in file_true.itertuples ():

            question1 = row.question1
            question2 = row.question2
            input_example = InputExample (texts = [question1, question2], label = 1.0)
            question_pair_package_true.append (input_example)

        question_pair_package_false = []
        for row in file_false.itertuples ():

            question1 = row.question1
            question2 = row.question2
            input_example = InputExample (texts = [question1, question2], label = 0.0)
            question_pair_package_false.append (input_example)

        triplet_package = []
        question_flatten = file['question1'].tolist () + file['question2'].tolist ()
        for row in file_true.itertuples ():
            anchor = row.question1  
            positive = row.question2  
            negative = random.choice (question_flatten)

            input_example = InputExample (texts = [anchor, positive, negative])
            triplet_package.append (input_example)

        merge_package_label = question_pair_package_true + question_pair_package_false

        return question_pair_package_true, merge_package_label, triplet_package
    
    def data_loader (self, question_pair_package_true, merge_package_label, triplet_package):

        batch_size = 32

        loader_mnrl = DataLoader (question_pair_package_true, shuffle = True, batch_size = batch_size)
        loader_csl = DataLoader (merge_package_label, shuffle = True, batch_size = batch_size)
        loader_tl = DataLoader (triplet_package, shuffle = True, batch_size = batch_size)
        loader_ll = DataLoader (merge_package_label, shuffle = True, batch_size = batch_size)

        return loader_mnrl, loader_csl, loader_tl, loader_ll 


if __name__ == '__main__':

    dataset = pd.read_csv ('quora-question-pairs/train.csv')
    train_test_split = joblib.load ('train_test_split.pkl')
    train_indices_list = train_test_split['train_dataset']
    test_indices_list = train_test_split['test_dataset']

    SBERT_Embedding_score = []
    for index in range (len (train_indices_list)):

        train_index = train_indices_list[index]
        test_index = test_indices_list[index]

        train_dataset = dataset.iloc[train_index]
        test_dataset = dataset.iloc[test_index]

        model = SBERT_Embedding ()
        model.fine_tune (train_dataset)
        cosine_similarity_score = model.predict (test_dataset)

        SBERT_Embedding_score.append (cosine_similarity_score)

    SBERT_Embedding_score = np.array (SBERT_Embedding_score)
    joblib.dump (SBERT_Embedding_score, 'SBERT_Embedding_Score.pkl')