import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors


#lastfm_diverse_tags_df = pd.read_csv(r"C:\Users\resha\data\lastfm_diverse_tags_df.csv")

lastfm_diverse_tags_df = pd.read_csv(r"C:\Users\corc4\data\lastfm_diverse_tags_df.csv")
#lastfm_pivot_df = pd.read_csv(r"C:\Users\resha\data\lastfm_pivot_df.csv")
#lastfm_tags_df = pd.read_csv(r"C:\Users\resha\data\lastfm_diverse_tags_df.csv")
track_metadata_df = pd.read_csv(r"C:\Users\corc4\data\track_metadata_df.csv")
train_triplets_df = pd.read_csv(r"C:\Users\corc4\data\train_triplets_df.csv")
play_count_grouped_df = pd.read_csv(r"C:\Users\corc4\data\play_count_grouped_df.csv")
genres_df = pd.read_csv(r"C:\Users\corc4\data\genres_df.csv")


track_metadata_df.columns
track_df = pd.merge(track_metadata_df, play_count_grouped_df, left_on='song_id', right_on='song').drop('song', axis=1)
track_df = pd.merge(track_df, genres_df, how='inner', on='track_id')
track_df = pd.merge(track_df, lastfm_diverse_pivot_df, how='left', left_on='track_id', right_on='tid').drop('tid', axis=1)


with pd.option_context('display.max_rows',None,):
    print(track_df.columns)





# pivot dataset
tag_counts = lastfm_diverse_tags_df.groupby('cleaned_tag')['tid'].nunique().sort_values(ascending=True).mean()

# Group by 'tid' and take the maximum 'tag_number' for each 'tid'
max_tag_number_per_tid = lastfm_diverse_tags_df.groupby('tid')['tag_number'].max().sort_values(ascending=True)

# Calculate the mean of these maximum tag numbers
mean_max_tag_number = max_tag_number_per_tid.mean()

lastfm_diverse_tags_df = lastfm_diverse_tags_df[lastfm_diverse_tags_df['tag_number'] >= 34]


combined_tags_df = lastfm_diverse_tags_df.groupby('tid')['cleaned_tag'].apply(lambda x: ' '.join([str(tag) for tag in x if pd.notna(tag)])).reset_index()

# there are no nulls
combined_tags_df["cleaned_tag"].isna().sum()

#Define a TF-IDF Vectorizer Object. Remove all english stop words such as 'the', 'a'
tfidf = TfidfVectorizer(stop_words='english')

#Construct the required TF-IDF matrix by fitting and transforming the data
tfidf_matrix = tfidf.fit_transform(combined_tags_df['cleaned_tag'])
tfidf_matrix = csr_matrix(tfidf_matrix)

knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=10, n_jobs=-1)
knn.fit(tfidf_matrix)


# Find the top 10 most similar items for each item
distances, indices = knn.kneighbors(tfidf_matrix)

# Convert distances to similarities
similarities = 1 - distances

# Create a sparse matrix for similarities
from scipy.sparse import lil_matrix

# Initialize an empty sparse matrix
cosine_sim_sparse = lil_matrix((tfidf_matrix.shape[0], tfidf_matrix.shape[0]))

# Populate the sparse matrix with the top similarities
for i, (similarity_values, index_values) in enumerate(zip(similarities, indices)):
    for similarity, index in zip(similarity_values, index_values):
        cosine_sim_sparse[i, index] = similarity

# Convert the matrix to CSR format for efficient storage and computation
cosine_sim_sparse = cosine_sim_sparse.tocsr()
















































############################ clustering tags ##################################

import torch
import torchtext
from torchtext.vocab import vocab

glove_model = api.load("glove-wiki-gigaword-300")
https://nlp.stanford.edu/pubs/glove.pdf
glove_model["you are a beautiful person"]

glove = torchtext.vocab.GloVe(name="6B", # trained on Wikipedia 2014 corpus
                              dim=100)    # embedding size = 50
glove_model['cat']
glove["cat"]
# pivoting tags 
lastfm_cleaned_tags_df = lastfm_diverse_tags_df.iloc[:,1:5]
#lastfm_cleaned_tags_df['tag_number'] = lastfm_tags_df.groupby('tid').cumcount() + 1
#lastfm_cleaned_pivot_df = lastfm_tags_df.pivot(index='tid', columns='tag_number', values='cleaned_tag').reset_index()
#lastfm_cleaned_pivot_df.columns = ['tid'] + [f'tag{i}' for i in range(1, len(lastfm_cleaned_pivot_df.columns))]

print(lastfm_cleaned_tags_df["cleaned_tag"].unique()[:50])

def get_tag_vector(tag, model):
    words = tag.split()
    vectors = [model[word] for word in words if word in model]
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        return None

tags = lastfm_pivot_df.iloc[:, 1:].stack().unique()  # Get unique tags, excluding NaNs

tag_vectors = {}

for tag in tags:
    if pd.notna(tag):
        vector = get_tag_vector(tag, glove_model)
        if vector is not None:
            tag_vectors[tag] = vector

def compute_song_vector(tags, tag_vectors):
    vectors = []
    for tag in tags:
        if tag in tag_vectors:
            vectors.append(tag_vectors[tag])
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(len(next(iter(tag_vectors.values()))))

lastfm_pivot_df['tags'] = lastfm_pivot_df.iloc[:, 1:].apply(lambda row: row.dropna().tolist(), axis=1)

lastfm_pivot_df['song_vector'] = lastfm_pivot_df['tags'].apply(lambda tags: compute_song_vector(tags, tag_vectors))

song_vectors = np.stack(lastfm_pivot_df['song_vector'].values)

similarity_matrix = cosine_similarity(song_vectors)

similarity_df = pd.DataFrame(similarity_matrix, index=lastfm_pivot_df['tid'], columns=lastfm_pivot_df['tid'])
print(similarity_df)

########################## getting genres ##################################

# get most popular tags:

tag_counts = lastfm_tags_df["tag"].value_counts()

for tag in tag_counts[:1000].index:
    print(tag)

tag_counts.to_csv('tags.csv', index=True)

lastfm_tags_df 

# unique tags:

len(lastfm_tags_df["tag"].unique()) # 505215 songs with 522366 unique tags

#"male", "female",""