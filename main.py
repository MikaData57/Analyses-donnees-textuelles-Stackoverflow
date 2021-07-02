# -*- coding: utf-8 -*-
"""
This main.py program gathers the functions necessary for the API entry point 
for the automatic creation of Tags of the questions on Stackoverflow.

The creation of the API is based on the FastAPI library.

|-----------------------------------------------------------------------------|
| Launch Uvicorn server : 
    - Go to app folder
    - Open cmd prompt with Python venv activate
    - type :  uvicorn main:app --reload
|-----------------------------------------------------------------------------|

|-----------------------------------------------------------------------------|
| Load documentation page with FastAPI : 
    - Open a web browser
    - Go to : http://127.0.0.1:8000/docs
|-----------------------------------------------------------------------------|

Created on Jul 2021
@author: Michael FUMERY
"""

# Import Python libraries
import os
from os.path import dirname, join, realpath, abspath
import joblib
import uvicorn
from fastapi import FastAPI
import numpy as np
import pandas as pd
import re
from bs4 import BeautifulSoup
from langdetect import detect
from deep_translator import GoogleTranslator
import nltk
from nltk.corpus import stopwords
import spacy

nltk.download('popular')

# Initialize FastAPI app
tags_metadata = [
    {"name": "Tags generator",
     "description": "The model tries to predict the tags associated with the \
         question asked. Just enter the text in the question \
             field to test the algorithm."}
    ]

app = FastAPI(
    title="Stackoverflow auto-tag Model API",
    description="Simple API using NLP methods to offer multiple Tags\
    for a given question with a pre-trained RandomForest model.",
    version="0.1",
    openapi_tags=tags_metadata)


# Load pre-trained TfidfVectorizer
with open(
        join(dirname(abspath("__file__")), 
             "models/tfidf_vectorizer.pkl"), "rb") as v_save:
    vectorizer = joblib.load(v_save)
    
# Load pre-trained Binarizer
with open(
        join(dirname(abspath("__file__")), 
             "models/multilabel_binarizer.pkl"), "rb") as b_save:
    multilabel_binarizer = joblib.load(b_save)
    
# Load pre-trained model
with open(
        join(dirname(abspath("__file__")), 
             "models/chain_randomforest_model.pkl"), "rb") as m_save:
    model = joblib.load(m_save)
    
# Cleaning function for new question
def remove_pos(nlp, x, pos_list):
    """NLP cleaning function based on the POS-Tagging of the Spacy library. 
    It allows you to keep only the Parts Of Speech listed as a parameter. 

    Parameters
    ----------------------------------------
    nlp : spacy pipeline
        Load pipeline with options.
        ex : spacy.load('en', exclude=['tok2vec', 'ner', 'parser', 
                                'attribute_ruler', 'lemmatizer'])
    x : string
        Sequence of characters to modify.
    pos_list : list
        List of POS to conserve.
    ----------------------------------------
    """
    # Test of language detection
    lang = detect(x)
    if(lang != "en"):
        # Deep translate if not in English
        x = GoogleTranslator(source='auto', target='en').translate(x)
    
    doc = nlp(x)
    list_text_row = []
    for token in doc:
        if(token.pos_ in pos_list):
            list_text_row.append(token.text)
    join_text_row = " ".join(list_text_row)
    join_text_row = join_text_row.lower().replace("c #", "c#")
    return join_text_row

def text_cleaner(x, nlp, pos_list, lang="english"):
    """Function allowing to carry out the preprossessing on the textual data. 
        It allows you to remove extra spaces, unicode characters, 
        English contractions, links, punctuation and numbers.
        
        The re library for using regular expressions must be loaded beforehand.
        The SpaCy and NLTK librairies must be loaded too. 

    Parameters
    ----------------------------------------
    nlp : spacy pipeline
        Load pipeline with options.
        ex : spacy.load('en', exclude=['tok2vec', 'ner', 'parser', 
                                'attribute_ruler', 'lemmatizer'])
    x : string
        Sequence of characters to modify.
    pos_list : list
        List of POS to conserve.
    ----------------------------------------
    """
    # Remove POS not in "NOUN", "PROPN"
    x = remove_pos(nlp, x, pos_list)
    # Case normalization
    x = x.lower()
    # Remove unicode characters
    x = x.encode("ascii", "ignore").decode()
    # Remove English contractions
    x = re.sub("\'\w+", '', x)
    # Remove ponctuation but not # (for C# for example)
    x = re.sub('[^\\w\\s#]', '', x)
    # Remove links
    x = re.sub(r'http*\S+', '', x)
    # Remove numbers
    x = re.sub(r'\w*\d+\w*', '', x)
    # Remove extra spaces
    x = re.sub('\s+', ' ', x)
        
    # Tokenization
    x = nltk.tokenize.word_tokenize(x)
    # List of stop words in select language from NLTK
    stop_words = stopwords.words(lang)
    # Remove stop words
    x = [word for word in x if word not in stop_words 
         and len(word)>2]
    # Lemmatizer
    wn = nltk.WordNetLemmatizer()
    x = [wn.lemmatize(word) for word in x]
    
    # Return cleaned text
    return x

# Create point of entrie for API
@app.get("/predict-tags", tags=["Tags generator"])
async def predict_tags(question: str):
    
    # Clean the question sent
    nlp = spacy.load('en', exclude=['tok2vec', 'ner', 'parser', 
                                'attribute_ruler', 'lemmatizer'])
    pos_list = ["NOUN","PROPN"]
    cleaned_question = text_cleaner(question, nlp, pos_list, "english")
    
    # Apply saved trained TfidfVectorizer
    X_tfidf = vectorizer.transform([cleaned_question])
    
    # Perform prediction
    predict = model.predict(X_tfidf)
    predict_probas = model.predict_proba(X_tfidf)
    # Inverse multilabel binarizer
    tags_predict = multilabel_binarizer.inverse_transform(predict)
    
    # DataFrame of probas
    df_predict_probas = pd.DataFrame(columns=['Tags', 'Probas'])
    df_predict_probas['Tags'] = multilabel_binarizer.classes_
    df_predict_probas['Probas'] = predict_probas.reshape(-1)
    # Select probas > 33%
    df_predict_probas = df_predict_probas[df_predict_probas['Probas']>=0.33]\
        .sort_values('Probas', ascending=False)
        
    # Results
    results = {}
    results['Tokens'] = cleaned_question
    results['Predicted Tags'] = tags_predict
    results['Predicted Tags Probabilities'] = df_predict_probas\
        .set_index('Tags')['Probas'].to_dict()
    
    return results

 
# Run app API with uvicorn   
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000, debug=True) 

#question = 'Ok, so I was looking through some data analysis (very basic) projects. I came across this line- print(df.groupby("level")["attempt"].mean()) Where df is the dataframe of the file https://raw.githubusercontent.com/whitehatjr/Data-Analysis-by-visualisation/master/data.csv Basically, as far as I can tell, It is the data of Grade 3 students who attempted a quiz, which had levels. Now, the only usages of groupby() I knew were - #First Usage q = df.groupby("") #Second Usage w = df.groupby(["", ""]) Can someone please explain to me, what the statement print(df.groupby("level")["attempt"].mean()) actually is?'
