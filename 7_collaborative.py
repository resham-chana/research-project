import pandas as pd
from sklearn.neighbors import NearestNeighbors


#https://github.com/d-elicio/Music-Recommender-System-from-scratch/blob/main/Music_Recommender_Project.ipynb

lastfm_diverse_tags_df = pd.read_csv(r"C:\Users\resha\data\lastfm_diverse_tags_df.csv")
lastfm_diverse_pivot_df = pd.read_csv(r"C:\Users\resha\data\lastfm_diverse_pivot_df.csv")
track_metadata_cleaned_df = pd.read_csv(r"C:\Users\resha\data\track_metadata_cleaned_df.csv")  
train_triplets_df = pd.read_csv(r"C:\Users\resha\data\train_triplets_df.csv")
play_count_grouped_df = pd.read_csv(r"C:\Users\resha\data\play_count_grouped_df.csv")
genres_df = pd.read_csv(r"C:\Users\resha\data\genres_df.csv")
track_features_all_df = pd.read_csv(r"C:\Users\resha\data\track_features_all_df.csv")


content_df = track_features_all_df[track_features_all_df['year'] != 0] \
    .drop_duplicates(subset=['title', 'artist_name']) \
    .dropna(subset=['cleaned_tag']) \
    .dropna(subset=['nationalities', 'gender', 'language', 'religion', 'continent'], how='all') \
    .rename(columns={'nationalities': 'country'}) \
    .reset_index(drop=True)

content_df.columns

train_triplets_df.columns

def min_max_normalize(series):
    return (series - series.min()) / (series.max() - series.min()) * 4 + 1

# Apply normalization to play_count
train_triplets_df['rating'] = min_max_normalize(train_triplets_df['play_count'])



min_songs = 20

# Calculate the number of unique songs each user has listened to
user_song_counts = train_triplets_df.groupby('user')['song'].nunique()

# Filter users based on the minimum number of songs listened to
users_to_keep = user_song_counts[user_song_counts > min_songs].index

# Create a new DataFrame with only the filtered users
filtered_train_triplets_df = train_triplets_df[train_triplets_df['user'].isin(users_to_keep)].reset_index(drop=True)

print(filtered_train_triplets_df)

# Drop the 'Unnamed: 0' column if not needed
user_ratings = train_triplets_df.drop(columns=['Unnamed: 0'])

user_ratings.sort_values(by='rating', ascending=False)

# https://campus.datacamp.com/courses/building-recommendation-engines-in-python/collaborative-filtering?ex=2

user_ratings_table = user_ratings.pivot(index='userId',columns='title',values='rating')






# Create a user-item matrix
user_item_matrix = df.pivot(index='user', columns='song', values='play_count').fillna(0)

# Train the KNN model
model_knn = NearestNeighbors(metric='cosine', algorithm='brute')
model_knn.fit(user_item_matrix)

# Function to get song recommendations for a user
def get_recommendations(user, n_neighbors=3):
    distances, indices = model_knn.kneighbors(user_item_matrix.loc[user].values.reshape(1, -1), n_neighbors=n_neighbors)
    
    recommendations = []
    for i in range(0, len(distances.flatten())):
        if i == 0:
            print(f'Recommendations for {user}:\n')
        else:
            recommendations.append(user_item_matrix.index[indices.flatten()[i]])
            print(f'{i}: {user_item_matrix.index[indices.flatten()[i]]}, with distance of {distances.flatten()[i]}')
    
    return recommendations

# Get recommendations for a specific user
user = 'user1'
recommendations = get_recommendations(user)




















track_features_df = pd.merge(track_metadata_cleaned_df, play_count_grouped_df.iloc[:,[1,2]], left_on='song_id', right_on='song').drop('song', axis=1)
track_features_df = pd.merge(track_features_df, genres_df.iloc[:,[1,2]], how='inner', on='track_id')
track_features_df = pd.merge(track_features_df, train_triplets_df, how='inner', on='track_id')
track_features_df.dropna(inplace= True)

tf_df = track_features_df.copy()

user_matrix = tf_df.pivot_table(index='user_id', 
                                    columns='title', values='play_count', fill_value=0)

sparsity = 1.0 - (((user_matrix == 0).sum().sum()) / float(user_matrix.size))

