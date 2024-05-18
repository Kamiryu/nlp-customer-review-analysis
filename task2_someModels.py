import pandas as pd
import numpy as np
from transformers import BertTokenizer, BertModel
import torch
from sklearn.cluster import KMeans
# from sklearn.cluster import DBSCAN
import string
import regex as re
from transformers import pipeline
import time 
from transformers import pipeline
import matplotlib.pyplot as plt





def classify_Kmeans(col_name='categories_list', seed_num=7):

    df_data_train = pd.read_parquet('data/ATML2024_Task2_PhiliBussRatings.parquet', engine='pyarrow')

    data = df_data_train[['business_id', 'categories_list']]
    data_train = data.explode('categories_list')
    data_train.drop_duplicates(subset=['categories_list'], keep='last', inplace=True)

    data_train['categories_list'] = data_train['categories_list'].astype(str).apply(lambda x: re.sub(f"[{string.punctuation}]", " ", x))

    # Load pre-trained BERT model and tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')

    # Function to get BERT embeddings
    def get_bert_embedding(text):
        inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

    # Get embeddings for each category
    data_train['col_embedding'] = data_train['categories_list'].astype(str).apply(lambda x: get_bert_embedding(x))


    # Prepare embeddings for clustering
    X = np.vstack(data_train['col_embedding'].values)

    # Perform K-Means clustering
    num_clusters = 7  # You can choose an appropriate number of clusters
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    data_train['cluster'] = kmeans.fit_predict(X)


    # Display the clusters
    for cluster in range(num_clusters):
        print(f"Cluster kmeans {cluster}:")
        print(data_train[data_train['cluster'] == cluster]['categories_list'].values)



def sentimentAnalysis(col_name):

    df_data_train = pd.read_parquet('data/ATML2024_Task2_PhiliBussRatings.parquet', engine='pyarrow')

    sent_pipe = pipeline("sentiment-analysis", # or "text-classification"
                        model="distilbert-base-uncased-finetuned-sst-2-english")

    df_data_train['textc1'] = df_data_train['text'].apply(lambda x: re.sub(f"[{string.punctuation}]", "", x.lower()))
    df_data_train['textc1'] = df_data_train['textc1'].apply(lambda x: x.replace('\n', ' '))

    df_data_train['textc1_len'] = df_data_train['textc1'].apply(lambda x: len(x))
    df_data_train['textc1_tok'] = df_data_train['textc1'].apply(lambda x: len(x.split(' ')))


    text_length = df_data_train['textc1'].apply(lambda x: len(x))
    text_length_token = df_data_train['textc1'].apply(lambda x: len(x.split(' ')))    
    _ = plt.hist(text_length, bins='auto')  # arguments are passed to np.histogram
    plt.title("Histogram with 'auto' bins")
    plt.show()

    _ = plt.hist(text_length_token, bins='auto')  # arguments are passed to np.histogram
    plt.title("Histogram with 'auto' bins")
    plt.show()

    df_data_train = df_data_train[ (df_data_train['textc1_len'] < 2500) & (df_data_train['textc1_tok'] < 500)]
    
    # sampled_df = df_data_train.sample(frac=1)

    print(df_data_train.shape)
    start = time.time()
    df_data_train['sent'] = df_data_train['textc1'].apply(sent_pipe)
    print(time.time() - start)


    df_data_train.to_parquet('data/ATML2024_Task2_ReviewsSentiment.parquet')
