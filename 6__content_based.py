# import libraries
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import OneHotEncoder
from sklearn.cluster import KMeans
import random
from sklearn.metrics import silhouette_score
import seaborn as sns
import matplotlib.pyplot as plt


# load in data
lastfm_diverse_tags_df = pd.read_csv(r"C:\Users\resha\data\lastfm_diverse_tags_df.csv")
track_features_all_df = pd.read_csv(r"C:\Users\resha\data\track_features_all_df.csv")

# check columns
track_features_all_df.columns

# pivot dataset
tag_counts = lastfm_diverse_tags_df.groupby('cleaned_tag')['tid'].nunique().sort_values(ascending=True).mean()

# check max tag number per track
max_tag_number_per_tid = lastfm_diverse_tags_df.groupby('tid')['tag_number'].max().sort_values(ascending=True)

# calculate the mean
mean_max_tag_number = max_tag_number_per_tid.mean()

# prune data even further - too much computational load in current state
lastfm_tags_pruned_df = lastfm_diverse_tags_df[lastfm_diverse_tags_df['tag_number'] >= 15]

# create one column of all tags combined for each track
combined_tags_df = lastfm_tags_pruned_df.groupby('tid')['cleaned_tag'].apply(lambda x: ' '.join([str(tag) for tag in x if pd.notna(tag)])).reset_index()

# there are no nulls
combined_tags_df["cleaned_tag"].isna().sum()

# merge combined tags
track_features_all_df = pd.merge(track_features_all_df, combined_tags_df[['tid', 'cleaned_tag']], 
                                  how='left', left_on='track_id', right_on='tid').drop('tid', axis=1)


# count diversity attributes
nationalities_counts = track_features_all_df['nationalities'].value_counts()
gender_counts = track_features_all_df['gender'].value_counts()
language_counts = track_features_all_df['language'].value_counts()
religion_counts = track_features_all_df['religion'].value_counts()
continent_counts = track_features_all_df['continent'].value_counts()

# print top 10
print(nationalities_counts.head(10))
print(gender_counts.head(10))
print(language_counts.head(10))
print(religion_counts.head(10))
print(continent_counts.head(10))

# turn columns to lowercase
track_features_all_df['title'] = track_features_all_df['title'].str.lower()
track_features_all_df['artist_name'] = track_features_all_df['artist_name'].str.lower()

# drop year = 0, duplicates and NaNs
content_df = track_features_all_df[track_features_all_df['year'] != 0] \
    .drop_duplicates(subset=['title', 'artist_name']) \
    .dropna(subset=['cleaned_tag']) \
    .dropna(subset=['nationalities', 'gender', 'language', 'religion', 'continent'], how='all') \
    .rename(columns={'nationalities': 'country'}) \
    .reset_index(drop=True)

# mutate for decade
content_df['decade'] = content_df['year'] - (content_df['year'] % 10)

# check there are no dupilcates
duplicates = content_df[content_df.duplicated(subset=['title', 'artist_name'], keep=False)]

content_df.to_csv(r"C:\Users\resha\data\content_df.csv")

########## tag based content recommender Using tdidf ##########

# code adapted from [https://www.datacamp.com/tutorial/recommender-systems-python]

# define a TF-IDF object and remove stop words
tfidf = TfidfVectorizer(stop_words='english')

# construct TF-IDF matrix by fitting/transforming data 
tfidf_matrix = tfidf.fit_transform(content_df['cleaned_tag'])

# calculate cosine similarity
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# create a map of indices and tracks
indices = pd.Series(content_df.index, index=content_df['track_id']).drop_duplicates()

def get_recommendations_tdidf(track_id, cosine_sim=cosine_sim):
    '''
    function to get 25 recommendations from one song 
    '''
    # index of the track
    idx = indices[track_id]

    # pairwsie similarity scores of all tracks with that track
    sim_scores = list(enumerate(cosine_sim[idx]))

    # sort tracks based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # grab the top 25
    sim_scores = sim_scores[1:25]

    track_indices = [i[0] for i in sim_scores]

    # return the 25 most similar tracks
    return content_df['track_id'].iloc[track_indices]

# get a random song from the data
track_id = content_df.sample(n=1).iloc[0]['track_id']
song = content_df[content_df['track_id'] == track_id][["title","artist_name"]]

# apply the recommender
recommendations_tdidf = pd.merge(get_recommendations_tdidf(track_id),
                           content_df[['track_id', 'title', 'artist_name' ,
                                       'country','gender', 'language',
                                        'religion', 'continent', 'genre','total_play_count','decade']], 
                           how = "left",on = "track_id")
print(recommendations_tdidf)


def calculate_diversity(recommendations_df, attributes):
    '''
    function to calculate how diverse the attributes are of the recommender
    '''
    diversity_scores = {}
    for attribute in attributes:
        # compute the number of unique languages/countries etc for each recommendation
        diversity_scores[attribute] = recommendations_df[attribute].nunique()
    return diversity_scores

# attributes to measure diversity:
attributes = ['gender', 'language', 'religion', 'continent', 'country','decade']

# apply function for 10 different songs (250 recommended songs)
calculate_diversity(recommendations_tdidf, attributes)
num_samples = 10
diversity_results_tdidf = []

for i in range(num_samples):
    track_id = content_df.sample(n=1).iloc[0]['track_id'] # pick a random song
    recommendations_tdidf = pd.merge(get_recommendations_tdidf(track_id), # apply recommender function
                               content_df[['track_id', 'title', 'country', 'artist_name', 'gender', 'language',
                                           'religion', 'continent', 'genre', 'total_play_count', 'decade']],
                               how="left", on="track_id")
    diversity_score_tdidf = calculate_diversity(recommendations_tdidf, attributes) # calculate diversity
    diversity_results_tdidf.append((track_id, diversity_score_tdidf))

for track_id, diversity_score_tdidf in diversity_results_tdidf:
    print(f"diversity score is {diversity_score_tdidf}")

avgerage_diversity_score_tdidf = {attribute: 0 for attribute in attributes}
for _, diversity_score_tdidf in diversity_results_tdidf:
    for attribute in attributes:
        avgerage_diversity_score_tdidf[attribute] += diversity_score_tdidf[attribute]

avgerage_diversity_score_tdidf = {attribute: score / num_samples for attribute, 
                            score in avgerage_diversity_score_tdidf.items()}
print(f"average diversity score: {avgerage_diversity_score_tdidf}")


########## tag based content recommender with K Means ##########

# adapted code from [https://www.datacamp.com/tutorial/k-means-clustering-python] and [https://www.datacamp.com/tutorial/recommender-systems-python]


# create the soup
def create_soup(row, main_factors):
    '''
    Function to join strings from columns together to create a 'soup'
    '''
    soup = ' '.join([str(row[col]) for col in main_factors])
    return soup

# main factors for similarity
main_factors = ['artist_name', 'cleaned_tag', 'genre']

# apply to content_df
content_df['soup'] = content_df.apply(create_soup, main_factors=main_factors, axis=1)

# create the count matrix and cosine similarity matrix
count = CountVectorizer(stop_words='english')
count_matrix = count.fit_transform(content_df['soup'])
cosine_sim2 = cosine_similarity(count_matrix, count_matrix)

# reset index and create a series for track_id indices
content_df = content_df.reset_index()
indices = pd.Series(content_df.index, index=content_df['track_id'])

# define diversity features
diversity_features = ['country', 'gender', 'language', 'religion', 'continent','decade']
df_diversity = content_df[diversity_features].fillna('Unknown')


# apply one hot encoding
encoder = OneHotEncoder()
encoded_features = encoder.fit_transform(df_diversity)

# k-means clustering with k=60
kmeans = KMeans(n_clusters=60, random_state=0)
content_df['cluster'] = kmeans.fit_predict(encoded_features)

K = range(2, 10)
fits = []
score = []

for k in K:
    # train the model for current value of k on training data
    model = KMeans(n_clusters = k, random_state = 0, n_init='auto').fit(encoded_features)
    fits.append(model)
    # append silhouette score to scores
    score.append(silhouette_score(encoded_features, model.labels_, metric='euclidean'))

# visualise silhouette score
plt.style.use("dark_background")
sns.lineplot(x=K, y=score, color='#4CAF50')
plt.title('Number of clusters vs. Silhouette Score')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Silhouette Score')
plt.savefig("sil_score.png")
plt.show()



# Recommendation function
def get_recommendations_kmeans(track_id, cosine_sim=cosine_sim2):
    '''
    function to return recommendations for a track with more diverse attributes
    '''
    idx = indices[track_id] 
    sim_scores = list(enumerate(cosine_sim[idx])) # get cosine similarity of tracks 
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True) # sort by most to least
    sim_scores = sim_scores[1:] # exclude original track

    recommended_indices = [i[0] for i in sim_scores]
    recommended_clusters = content_df['cluster'].iloc[recommended_indices].values
    
    # select tracks from different clusters
    unique_clusters = list(set(recommended_clusters))
    final_recommendations = []

    # loop through each cluster and get indices of tracks from current cluster
    for cluster in unique_clusters:
        cluster_indices = [i for i in recommended_indices if content_df['cluster'].iloc[i] == cluster]
        if cluster_indices:
            final_recommendations.append(random.choice(cluster_indices))
        if len(final_recommendations) >= 25: # select only 25 songs
            break

    return content_df['track_id'].iloc[final_recommendations]

# get random song 
track_id = content_df.sample(n=1).iloc[0]['track_id']
song = content_df[content_df['track_id'] == track_id][["title","artist_name"]]

# apply recommendation algoithm to it
recommendations_kmeans = pd.merge(get_recommendations_kmeans(track_id),
                           content_df[['track_id', 'title', 'artist_name' , 'country','gender', 'language',
                                        'religion', 'continent', 'genre','total_play_count','decade']], 
                           how = "left",on = "track_id")
print(recommendations_kmeans)


attributes = ['gender', 'language', 'religion', 'continent', 'country','decade']

# calculate diversity for 10 sample songs (250 songs total)
calculate_diversity(get_recommendations_kmeans, attributes)
num_samples = 10
diversity_results_kmeans = []

for i in range(num_samples):
    track_id = content_df.sample(n=1).iloc[0]['track_id'] 
    recommendations_kmeans = pd.merge(get_recommendations_kmeans(track_id),
                               content_df[['track_id', 'title', 'country', 'artist_name', 'gender', 'language',
                                           'religion', 'continent', 'genre', 'total_play_count', 'decade']],
                               how="left", on="track_id")
    diversity_score_kmeans = calculate_diversity(recommendations_kmeans, attributes) 
    diversity_results_kmeans.append((track_id, diversity_score_kmeans))

# print 
for track_id, diversity_score_kmeans in diversity_results_kmeans:
    print(f"diversity score is (k-means) {diversity_score_kmeans}")

# get average of all recommendations
avgerage_diversity_score_kmeans = {attribute: 0 for attribute in attributes}
for _, diversity_score in diversity_results_kmeans:
    for attribute in attributes:
        avgerage_diversity_score_kmeans[attribute] += diversity_score[attribute]

avgerage_diversity_score_kmeans = {attribute: score / num_samples for attribute, 
                            score in avgerage_diversity_score_kmeans.items()}
print(f"average diversity score: {avgerage_diversity_score_kmeans}")