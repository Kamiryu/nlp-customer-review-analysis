import pandas as pd
import ast as ast
import numpy as np
import string
from textblob import TextBlob
# string.punctuation

from typing import *

########################################################################################################################################################
# sentiment analysis
from transformers import pipeline
def sentiment(text):
    output = sent_pipe(text)[0]['label']
    return output

########################################################################################################################################################

def extract_weekend_value(hours_dictionary:Dict) -> bool:
    days =  hours_dictionary.keys()
    if 'Saturday' in days  or 'Sunday' in days :
        return True
    return False

def extract_hours_value(hours_dictionary:Dict) -> List:
    values = dict(zip(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'], list([-1, -1]  for i in range (7))))
    
    for (k,v) in hours_dictionary.items():
        vals = v.split('-')
        vals_int = [ int(x.split(':')[0]) + (int(x.split(':')[1])/60) for x in vals]
        values[k] = vals_int

    return list(values.values())


def extract_attributes_value(attr_dictionary: Dict) -> List:
    values = []
    for (k, v) in attr_dictionary.items():
        
        if isinstance(v, dict):
            for (k2, v2) in v.items():
                values.append(k2)
        elif v:
            values.append(k)

        
    return values



def extract_attributes_value2(attr_dictionary: Dict) -> List:
    values = []
    for (k, v) in attr_dictionary.items():

        try: 
            if isinstance(ast.literal_eval(v), dict):
                for (k2, v2) in ast.literal_eval(v).items():                    
                    if v2:
                        # print(k+"-"+k2)
                        values.append( (k+"-"+k2).lower() )
            elif v:
                values.append(k.lower())
        except:
            if v:
                values.append(k.lower())
    return values

########################################################################################################################################################
import nltk
# from nltk.stem import PorterStemmer
# from nltk.stem.snowball import SnowballStemmer

stopwords = nltk.corpus.stopwords.words("english")
wnl = nltk.stem.WordNetLemmatizer()


def custom_tokenizer(text: str, pos_arg={"VERB":'v', "ADJ":'a', "ADV":'r', "NOUN":'n'}) -> List:
    """
    Accepts a text as a string, tokenize and lemmatize it
    """
    output = []
    text_tokens = nltk.word_tokenize(text)
    
    for w, pos in nltk.pos_tag(text_tokens, tagset="universal"):
        pos = pos_arg[pos] if pos in pos_arg.keys() else 'n'
        lem = wnl.lemmatize(w, pos=pos)

        if lem not in stopwords:
            output.append(lem)

    # print the original tokens and their lemma
    # print(f'{w} -> {lem}')
  
    return output


########################################################################################################################################################
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer


def count_word_vector(df : pd.DataFrame, column_name: str, lowercase=True) -> List[List]:
    """
    Returns bag-of-words representation of a given column in a dataframe.
    It uses the custom tokenizer defined above. 
    """
    vectorizer = CountVectorizer(tokenizer=custom_tokenizer ,lowercase=lowercase) # min_df=1e-3 to ignore unfrequent words
    bow_repr = vectorizer.fit_transform(df['column_name'])
    
    return bow_repr


def tfidf_representation(bow_representation: List[List]) -> List[List]:
    tfidf_transformer = TfidfTransformer()
    tfidf_repr = tfidf_transformer.fit_transform(bow_representation)

    return tfidf_repr


######################## ASPECT BASED SENTIMENT ANALYSIS ##############################
#######################################################################################
def extract_aspect_sentences(review, aspect_keywords):
    aspect_sentences = {aspect: [] for aspect in aspect_keywords}
    sentences = nltk.tokenize.sent_tokenize(review)
    
    for sentence in sentences:
        for aspect, keywords in aspect_keywords.items():
            if any(keyword in sentence.lower() for keyword in keywords):
                aspect_sentences[aspect].append(sentence)
    
    return aspect_sentences

def get_sentiment(text):
    blob = TextBlob(text)
    # Determine sentiment polarity
    sentiment_polarity = blob.sentiment.polarity
    # Classify sentiment as positive, negative, or neutral
    if sentiment_polarity > 0:
        return 'positive'
    elif sentiment_polarity < 0:
        return 'negative'
    else:
        return 'neutral'

def analyze_aspects(review, aspect_keywords):
    aspect_sentences = extract_aspect_sentences(review, aspect_keywords)
    aspect_sentiments = {}
    
    for aspect, sentences in aspect_sentences.items():
        sentiments = [get_sentiment(sentence) for sentence in sentences]
        # Aggregate sentiment for the aspect
        if sentiments:
            sentiment = max(set(sentiments), key=sentiments.count)
        else:
            sentiment = 'neutral'
        aspect_sentiments[aspect] = sentiment
    
    return aspect_sentiments


