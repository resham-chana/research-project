import pandas as pd
from sklearn.calibration import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors


#lastfm_diverse_tags_df = pd.read_csv(r"C:\Users\resha\data\lastfm_diverse_tags_df.csv")

lastfm_diverse_tags_df = pd.read_csv(r"C:\Users\resha\data\lastfm_diverse_tags_df.csv")
lastfm_diverse_pivot_df = pd.read_csv(r"C:\Users\resha\data\lastfm_diverse_pivot_df.csv")
#lastfm_tags_df = pd.read_csv(r"C:\Users\resha\data\lastfm_diverse_tags_df.csv")
#track_metadata_cleaned_df = pd.read_csv(r"C:\Users\resha\data\track_metadata_cleaned_df.csv")
#train_triplets_df = pd.read_csv(r"C:\Users\corc4\data\train_triplets_df.csv")
#play_count_grouped_df = pd.read_csv(r"C:\Users\corc4\data\play_count_grouped_df.csv")
#genres_df = pd.read_csv(r"C:\Users\resha\data\genres_df.csv")
#MSD_merged_df = pd.read_csv(r"C:\Users\resha\data\MSD_merged_df.csv")
track_features_all_df = pd.read_csv(r"C:\Users\resha\data\track_features_all_df.csv")

track_features_all_df.columns
#track_metadata_cleaned_df.columns
#MSD_merged_df.columns
#track_metadata_df.columns
#track_df = pd.merge(track_metadata_df, play_count_grouped_df, left_on='song_id', right_on='song').drop('song', axis=1)
#track_df = pd.merge(track_df, genres_df, how='inner', on='track_id')
#track_df = pd.merge(track_df, lastfm_diverse_pivot_df, how='left', left_on='track_id', right_on='tid').drop('tid', axis=1)


#with pd.option_context('display.max_rows',None,):
#    print(track_df.columns)

# pivot dataset
tag_counts = lastfm_diverse_tags_df.groupby('cleaned_tag')['tid'].nunique().sort_values(ascending=True).mean()

# Group by 'tid' and take the maximum 'tag_number' for each 'tid'
max_tag_number_per_tid = lastfm_diverse_tags_df.groupby('tid')['tag_number'].max().sort_values(ascending=True)

# Calculate the mean of these maximum tag numbers
mean_max_tag_number = max_tag_number_per_tid.mean()

lastfm_tags_pruned_df = lastfm_diverse_tags_df[lastfm_diverse_tags_df['tag_number'] >= 15]


combined_tags_df = lastfm_tags_pruned_df.groupby('tid')['cleaned_tag'].apply(lambda x: ' '.join([str(tag) for tag in x if pd.notna(tag)])).reset_index()

# there are no nulls
combined_tags_df["cleaned_tag"].isna().sum()

#final_features = track_features_df.merge(MSD_merged_df[['track_id', 'loudness', 'country', 'mode', 'key', 'energy', 'tempo']], on='track_id', how='left')

#content_df = pd.merge(final_features, combined_tags_df, left_on='track_id', right_on='tid').drop('tid', axis=1)

track_features_all_df = pd.merge(track_features_all_df, combined_tags_df[['tid', 'cleaned_tag']], 
                                  how='left', left_on='track_id', right_on='tid').drop('tid', axis=1)

track_features_all_df.columns

# Generate value counts for each specified column
nationalities_counts = track_features_all_df['nationalities'].value_counts()
gender_counts = track_features_all_df['gender'].value_counts()
language_counts = track_features_all_df['language'].value_counts()
religion_counts = track_features_all_df['religion'].value_counts()
continent_counts = track_features_all_df['continent'].value_counts()

# Print the results
print("Top 10 Nationalities:")
print(nationalities_counts.head(10))
print("\nTop 10 Genders:")
print(gender_counts.head(10))
print("\nTop 10 Languages:")
print(language_counts.head(10))
print("\nTop 10 Religions:")
print(religion_counts.head(10))
print("\nTop 10 Continents:")
print(continent_counts.head(10))


track_features_all_df.columns



content_df = track_features_all_df[track_features_all_df['year'] != 0] \
    .drop_duplicates(subset=['title', 'artist_name']) \
    .dropna(subset=['cleaned_tag']) \
    .dropna(subset=['nationalities', 'gender', 'language', 'religion', 'continent'], how='all') \
    .reset_index(drop=True)

duplicates = content_df[content_df.duplicated(subset=['title', 'artist_name'], keep=False)]


#Define a TF-IDF Vectorizer Object. Remove all english stop words such as 'the', 'a'
tfidf = TfidfVectorizer(stop_words='english')

#Construct the required TF-IDF matrix by fitting and transforming the data
tfidf_matrix = tfidf.fit_transform(content_df['cleaned_tag'])

cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

indices = pd.Series(content_df.index, index=content_df['title']).drop_duplicates()

def get_recommendations(title, feature, cosine_sim=cosine_sim, top_n=10):
    # Check if the title is in the DataFrame
    if title not in indices:
        return "Title not found in database."

    # Get the index of the track that matches the title
    idx = indices[title]

    # Get the pairwise similarity scores of all tracks with that track
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the tracks based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the top N most similar tracks
    sim_scores = sim_scores[1:top_n+1]

    # Get the track indices
    track_indices = [i[0] for i in sim_scores]

    # Retrieve the top N most similar tracks
    recommended_tracks = content_df.iloc[track_indices]

    # Diversity scoring
    def diversity_score(row, feature):
        if pd.isna(row[feature]):
            return 0
        return row[feature]  # You might want to normalize or adjust this score based on feature

    # Normalize features
    for feature in ['year', 'total_play_count', 'genre', 'play_count', 'nationalities', 'gender', 'language', 'religion', 'continent']:
        if feature not in content_df.columns:
            continue
        if content_df[feature].dtype == 'object':
            le = LabelEncoder()
            content_df[feature] = le.fit_transform(content_df[feature].astype(str))

    # Apply diversity scoring
    recommended_tracks['diversity_score'] = recommended_tracks.apply(lambda row: diversity_score(row, feature), axis=1)

    # Sort by diversity score
    recommended_tracks = recommended_tracks.sort_values(by='diversity_score', ascending=False)

    # Return the top N tracks
    return recommended_tracks[['title', 'artist_name', 'genre', 'language', 'nationalities', feature]].head(top_n)


title = content_df.sample(n=1).iloc[0]['title']
  # Replace with the title of the song you want recommendations for
feature = 'continent'  # Replace with the feature you want to use for diversity

recommendations = get_recommendations(title, feature)
print(recommendations)















def get_recommendations(title, feature, cosine_sim=cosine_sim, top_n=10):
    """
    Get recommendations based on cosine similarity and diversity scoring.
    
    :param title: The title of the track for which recommendations are to be made.
    :param feature: The feature used for diversity scoring.
    :param cosine_sim: Precomputed cosine similarity matrix.
    :param top_n: Number of top recommendations to return.
    :return: DataFrame with recommended tracks.
    """
    if title not in indices:
        return pd.DataFrame()  # Return empty DataFrame if title not found

    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n+1]
    track_indices = [i[0] for i in sim_scores]
    
    recommended_tracks = content_df.iloc[track_indices].copy()  # Make a copy to avoid warnings

    # Check if the feature exists in the DataFrame
    if feature not in content_df.columns:
        return recommended_tracks

    # Encode categorical features if necessary
    if content_df[feature].dtype == 'object':
        le = LabelEncoder()
        content_df[feature] = le.fit_transform(content_df[feature].astype(str))
    
    def diversity_score(row, feature):
        if pd.isna(row[feature]):
            return 0
        return row[feature]  # Consider normalization if needed

    # Calculate diversity scores
    recommended_tracks.loc[:, 'diversity_score'] = recommended_tracks.apply(lambda row: diversity_score(row, feature), axis=1)

    # Sort by diversity score
    recommended_tracks = recommended_tracks.sort_values(by='diversity_score', ascending=False)
    
    return recommended_tracks[['title', 'artist_name', 'genre', 'language', 'nationalities', feature, 'diversity_score']].head(top_n)


title = content_df.sample(n=1).iloc[0]['title']
  # Replace with the title of the song you want recommendations for
feature = 'continent'  # Replace with the feature you want to use for diversity

recommendations = get_recommendations(title, feature)
print(recommendations)



def precision_at_k(recommendations, relevant_items, k=10):
    # Assume relevant_items is a set of relevant item titles
    top_k_recommendations = set(recommendations.head(k)['title'])
    relevant_items_set = set(relevant_items)
    intersection = top_k_recommendations.intersection(relevant_items_set)
    return len(intersection) / k


# Example usage
relevant_items = content_df[content_df['title'] == title]['title'].tolist()  # Placeholder, update with actual relevant items
recommendations = get_recommendations(title, feature)
precision = precision_at_k(recommendations, relevant_items)
print(f"Precision@10: {precision:.2f}")

def mean_average_precision(df, feature, k=10):
    """
    Compute Mean Average Precision (MAP) over all unique titles in the DataFrame.
    
    :param df: DataFrame containing the track features.
    :param feature: Feature used for diversity scoring.
    :param k: Number of top recommendations to consider.
    :return: Mean Average Precision.
    """
    queries = df['title'].unique()
    average_precisions = []
    
    for title in queries:
        relevant_items = df[df['title'] == title]['title'].tolist()
        recommendations = get_recommendations(title, feature)
        
        # Check if recommendations DataFrame is empty
        if recommendations.empty:
            continue
        
        precision = precision_at_k(recommendations, relevant_items, k)
        average_precisions.append(precision)
    
    return sum(average_precisions) / len(average_precisions) if average_precisions else 0

# Example usage
map_score = mean_average_precision(content_df, 'genre')
print(f"Mean Average Precision: {map_score:.2f}")













def get_recommendations(title, cosine_sim=cosine_sim, top_n=10):
    # Check if the title is in the DataFrame
    if title not in indices:
        return "Title not found in database."

    # Get the index of the track that matches the title
    idx = indices[title]

    # Get the pairwise similarity scores of all tracks with that track
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the tracks based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the top N most similar tracks
    sim_scores = sim_scores[1:top_n+1]

    # Get the track indices
    track_indices = [i[0] for i in sim_scores]

    # Retrieve the top N most similar tracks
    recommended_tracks = content_df.iloc[track_indices]

    # Rank recommendations based on language and nationality
    def rank_score(row):
        score = 0
        if pd.notna(row['language']) and row['language'] != 'English':
            score += 1  # Adjust weight as necessary
        if pd.notna(row['nationalities']) and row['nationalities'] != 'United States':
            score += 1  # Adjust weight as necessary
        return score

    # Apply ranking score
    recommended_tracks['rank_score'] = recommended_tracks.apply(rank_score, axis=1)

    # Sort the recommended tracks based on the rank score
    recommended_tracks = recommended_tracks.sort_values(by='rank_score', ascending=False)

    # Return the top N tracks
    return recommended_tracks[['title', 'artist_name', 'genre', 'language', 'nationalities']].head(top_n)











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