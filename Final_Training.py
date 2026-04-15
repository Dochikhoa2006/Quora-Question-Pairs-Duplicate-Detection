from SBERT_embedding import SBERT_Embedding
from LightGBM_lexical import LightGBM_Lexical
import pandas as pd
import joblib

dataset = pd.read_csv ('/Users/chikhoado/Desktop/PROJECTS/Quora Question/quora-question-pairs/train.csv')

SBERT = SBERT_Embedding ()
SBERT.fine_tune (dataset)

LightGBM = LightGBM_Lexical (dataset)
LightGBM.train (dataset)

SBERT.save_model ()
joblib.dump (LightGBM, 'LightGBM_model.pkl')





# cd '/Users/chikhoado/Desktop/PROJECTS/Quora Question'
# /opt/homebrew/bin/python3.12 -m venv .venv
# source .venv/bin/activate
# pip install pandas 
# python '/Users/chikhoado/Desktop/PROJECTS/Quora Question/Final_Training.py'