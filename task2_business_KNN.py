import pandas as pd
import numpy as np
from transformers import BertTokenizer, BertModel
import torch
from sklearn.cluster import 

df_data = pd.read_parquet('data/ATML2024_Task2_PhiliBussRatings.parquet', engine='pyarrow')
business_cat = np.concatenate(df_data.categories_list.values).ravel()



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
reviews['category_embedding'] = reviews['category'].apply(lambda x: get_bert_embedding(x))

# Prepare embeddings for clustering
X = np.vstack(reviews['category_embedding'].values)

# Perform K-Means clustering
num_clusters = 5  # You can choose an appropriate number of clusters
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
reviews['cluster'] = kmeans.fit_predict(X)

# Display the clusters
for cluster in range(num_clusters):
    print(f"Cluster {cluster}:")
    print(reviews[reviews['cluster'] == cluster]['category'].values)
