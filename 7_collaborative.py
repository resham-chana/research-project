import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor, NearestNeighbors
import sklearn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import svds
import scipy.sparse as sp


#https://github.com/d-elicio/Music-Recommender-System-from-scratch/blob/main/Music_Recommender_Project.ipynb

lastfm_diverse_tags_df = pd.read_csv(r"C:\Users\resha\data\lastfm_diverse_tags_df.csv")
lastfm_diverse_pivot_df = pd.read_csv(r"C:\Users\resha\data\lastfm_diverse_pivot_df.csv")
track_metadata_cleaned_df = pd.read_csv(r"C:\Users\resha\data\track_metadata_cleaned_df.csv")  
train_triplets_df = pd.read_csv(r"C:\Users\resha\data\train_triplets_df.csv")
play_count_grouped_df = pd.read_csv(r"C:\Users\resha\data\play_count_grouped_df.csv")
genres_df = pd.read_csv(r"C:\Users\resha\data\genres_df.csv")
track_features_all_df = pd.read_csv(r"C:\Users\resha\data\track_features_all_df.csv")
content_df = pd.read_csv(r"C:\Users\resha\data\content_df.csv")


collab_df = pd.merge(content_df, train_triplets_df, left_on="song_id", right_on="song", how="left")
collab_df = collab_df.dropna(subset=['user'])
collab_df = collab_df.drop(["Unnamed: 0.2", "Unnamed: 0.1","Unnamed: 0_x",
                            'Unnamed: 0_y',"song","title","song_id","release",
                            "artist_id",'artist_mbid',
                            'artist_name', 'duration', 'artist_familiarity', 'artist_hotttnesss','year','genre', 
                            'log_total_play_count', 'country',
                            'gender', 'language', 'religion', 'continent', 'cleaned_tag', 'decade'], axis=1)
collab_df.head(50)
collab_df.sort_values(by="play_count")
collab_df["log_play_count"]= np.log(collab_df["play_count"])


############################### outlier removal with IQR #################################################
#https://www.geeksforgeeks.org/detect-and-remove-the-outliers-using-python/

play_count_values = collab_df['play_count'].value_counts()
collab_df["play_count"].mean()
collab_df["total_play_count"].max()

test = collab_df.copy()

# IQR
# Calculate the upper and lower limits
Q1 = test['total_play_count'].quantile(0.25)
Q3 = test['total_play_count'].quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5*IQR
upper = Q3 + 1.5*IQR

#filtering songs only listened to less than 3 times 
collab_df['play_count'].mean() # this is 3
collab_df = collab_df[(collab_df['play_count'] >= 3)]

collab_df.sort_values(by="play_count", ascending=True).head(10)
collab_df.sort_values(by="total_play_count", ascending=True).head(10)

# only include users that have listened to 10 songs or more 
user_song_counts = collab_df.groupby('user')['track_id'].nunique()
user_song_counts.mean() # this is 1
users_more_than_10_songs = user_song_counts[user_song_counts > 1].index

# Filter the DataFrame to include only users who have listened to more than 10 songs
collab_df = collab_df[collab_df['user'].isin(users_more_than_10_songs)]

# Show the shape of the new DataFrame to verify
print("New Shape: ", collab_df.shape)

# Display the first few rows to verify the data
collab_df.head()

collab_df

# normalise

def min_max_normalize(series):
    return (series - series.min()) / (series.max() - series.min()) * 4 + 1

# Apply normalization to play_count
collab_df['rating'] = min_max_normalize(collab_df['play_count'])

collab_df['rating'].plot(kind='hist', bins=1000, figsize=(10, 6))
plt.show()

######################################### collaborative filtering system ############################
# https://campus.datacamp.com/courses/building-recommendation-engines-in-python/collaborative-filtering?ex=2


user_ratings_df = collab_df.pivot(index='user',columns='track_id',values='rating')


# Count the occupied cells
sparsity_count = user_ratings_df.isnull().values.sum()

# Count all cells
full_count = user_ratings_df.size

# Find the sparsity of the DataFrame
sparsity = sparsity_count / full_count
print(sparsity)


# Get the average rating for each user 
avg_ratings = user_ratings_df.mean(axis=1)
# Center each users ratings around 0
user_ratings_centered_df = user_ratings_df.sub(avg_ratings, axis=0)
# Fill in the missing data with 0s
user_ratings_centered_df.fillna(0, inplace=True)

user_ratings_sparse = sp.csr_matrix(user_ratings_centered_df.values)

#user_ratings_centered_df = user_ratings_centered_df.reset_index(drop=True)
# Decompose the matrix
U, sigma, Vt = svds(user_ratings_sparse)

# Convert sigma into a diagonal matrix
sigma = np.diag(sigma)
print(sigma)


# Dot product of U and sigma
U_sigma = np.dot(U, sigma)

# Dot product of result and Vt
U_sigma_Vt = np.dot(U_sigma, Vt)

# Add back on the row means contained in avg_ratings
uncentered_ratings = U_sigma_Vt + avg_ratings.values.reshape(-1, 1)

# Create DataFrame of the results
calc_pred_ratings_df = pd.DataFrame(uncentered_ratings, 
                                    index=user_ratings_df.index,
                                    columns=user_ratings_df.columns
                                   )
# Print both the recalculated matrix and the original 
print(calc_pred_ratings_df)
print(user_ratings_df)



# Sort the ratings of User 5 from high to low
user_example_ratings = calc_pred_ratings_df.loc['fff9bd021bf6e07936883b9bb045207fcf372a2c',:].sort_values(ascending=False)

print(user_example_ratings)


############################################# evaluate ####################################################



# Split the data into training and test sets
flattened_df = user_ratings_df.stack().reset_index()
flattened_df.columns = ['user', 'track_id', 'rating']
train_data, test_data = train_test_split(flattened_df, test_size=0.1, random_state=42)

# Pivot back to the original DataFrame format for train and test sets
train_df = train_data.pivot(index='user', columns='track_id', values='rating')
test_df = test_data.pivot(index='user', columns='track_id', values='rating')

# Ensure the test set has the same structure as the predicted ratings
test_df = test_df.reindex(index=calc_pred_ratings_df.index, columns=calc_pred_ratings_df.columns)
test_df.fillna(0, inplace=True)

# Calculate RMSE on the test set
actual_values = test_df.values
predicted_values = calc_pred_ratings_df.values
mask = ~np.isnan(actual_values)

rmse = mean_squared_error(actual_values[mask], predicted_values[mask], squared=False)
print(f"RMSE: {rmse}")

# Example of user-specific ratings
user_example_ratings = calc_pred_ratings_df.loc['specific_user'].sort_values(ascending=False)
print(user_example_ratings)



































user_ratings.sort_values(by='rating', ascending=False)

# https://campus.datacamp.com/courses/building-recommendation-engines-in-python/collaborative-filtering?ex=2

user_ratings_table = user_ratings.pivot(index='userId',columns='title',values='rating')







# Count the occupied cells
sparsity_count = user_ratings_df.isnull().values.sum()

# Count all cells
full_count = user_ratings_df.size

# Find the sparsity of the DataFrame
sparsity = sparsity_count / full_count
print(sparsity)













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

