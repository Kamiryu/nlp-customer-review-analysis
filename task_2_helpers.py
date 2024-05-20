import pandas as pd
import ast as ast
import numpy as np
import string
from typing import *
pd.options.mode.chained_assignment = None 


########################################################################################################################################################

def split_tokens(df_data: pd.DataFrame) -> pd.DataFrame:
    # Find quantity of reviews with more than 375 tokens
    df_data["token"] = df_data["text"].apply(lambda x: len(x.split(" ")))
    df_data[df_data["token"] >= 375].shape[0] / df_data.shape[0]
    # print(df_data["token"].describe())

    # Split the dataset up to 3 parts of 125 tokens each
    df_1 = df_data[(df_data["token"] >= 1) & (df_data["token"] <= 125)]
    df_2 = df_data[(df_data["token"] >= 126) & (df_data["token"] <= 250)]
    df_3 = df_data[(df_data["token"] >= 251) & (df_data["token"] <= 375)]

    # Write in new columns the tokens
    df_2["token_1"] = df_2["text"].apply(lambda x: " ".join(x.split()[:125]))
    df_2["token_2"] = df_2.loc["text"].apply(lambda x: " ".join(x.split()[125:250]))

    df_3["token_1"] = df_3["text"].apply(lambda x: " ".join(x.split()[:125]))
    df_3["token_2"] = df_3["text"].apply(lambda x: " ".join(x.split()[125:250]))
    df_3["token_3"] = df_3["text"].apply(lambda x: " ".join(x.split()[250:375]))

    return  df_1, df_2, df_3








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

        if v:
            values.append(k)
     
    return values


def extract_attributes_value_nested(attr_dictionary: Dict) -> List:
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



def select_preprocess_Phili_business(df_business: pd.DataFrame) -> pd.DataFrame:
    df_business_ph = df_business.copy()
    df_business_ph['postal_code_int'] = pd.to_numeric(df_business_ph['postal_code'], errors='coerce', downcast='integer')
    df_business_ph = df_business_ph.loc[ (df_business_ph['postal_code_int'] >= 19019) & (df_business_ph['postal_code_int'] <= 19255) ]

    # Convert attributes
    df_business_ph['attributes'] = df_business_ph['attributes'].apply(lambda x: ast.literal_eval(x) if not pd.isna(x) or not pd.isnull(x)  else dict())
    df_business_ph['attributes_list'] = df_business_ph['attributes'].apply(lambda x: sorted(extract_attributes_value_nested(x)))

    # Convert hours
    df_business_ph['hours'] = df_business_ph['hours'].apply(lambda x: ast.literal_eval(x) if not pd.isna(x) or not pd.isnull(x)  else dict())
    df_business_ph['hours_list'] = df_business_ph['hours'].apply(lambda x: extract_hours_value(x))

    # Convert categories
    df_business_ph['categories_list'] = df_business_ph['categories'].apply(lambda x: sorted(map(lambda a: a.strip().lower(), x.split(',')) ) if isinstance(x, str) else [x])

    return df_business_ph



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
