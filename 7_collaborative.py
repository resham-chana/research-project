import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor, NearestNeighbors
import sklearn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import svds
import scipy.sparse as sp
from sklearn.metrics import r2_score


#https://github.com/d-elicio/Music-Recommender-System-from-scratch/blob/main/Music_Recommender_Project.ipynb

#lastfm_diverse_tags_df = pd.read_csv(r"C:\Users\resha\data\lastfm_diverse_tags_df.csv")
#lastfm_diverse_pivot_df = pd.read_csv(r"C:\Users\resha\data\lastfm_diverse_pivot_df.csv")
#track_metadata_cleaned_df = pd.read_csv(r"C:\Users\resha\data\track_metadata_cleaned_df.csv")  
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
users_more_than_10_songs = user_song_counts[user_song_counts >= 2].index

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

######################################### collaborative filtering system with matrix factorisation ############################
# https://campus.datacamp.com/courses/building-recommendation-engines-in-python/collaborative-filtering?ex=2

user_ratings_df = collab_df.pivot(index='user',columns='track_id',values='rating')

# Count the occupied cells
sparsity_count = user_ratings_df.isnull().values.sum()

# Count all cells
full_count = user_ratings_df.size

# Find the sparsity of the DataFrame
sparsity = sparsity_count / full_count
print(sparsity)

# Flatten the DataFrame into a list of (user, track_id, rating) tuples
ratings = user_ratings_df.stack().reset_index()
ratings.columns = ['user', 'track_id', 'rating']

# Split into train and test sets
train_data, test_data = train_test_split(ratings, test_size=0.2, random_state=42)

# Convert train and test sets back to pivot format (so they match the original user_ratings_df structure)
train_df = train_data.pivot(index='user', columns='track_id', values='rating')
test_df = test_data.pivot(index='user', columns='track_id', values='rating')

# Step 1: Preprocessing the Training Data
# Get the average rating for each user in the training set
avg_ratings = train_df.mean(axis=1)
# Center each user's ratings around 0
user_ratings_centered_train_df = train_df.sub(avg_ratings, axis=0)
# Fill in the missing data with 0s
user_ratings_centered_train_df.fillna(0, inplace=True)
# Convert the centered ratings DataFrame to a sparse matrix
user_ratings_sparse = sp.csr_matrix(user_ratings_centered_train_df.values)

# Perform matrix factorization (SVD)
U, sigma, Vt = svds(user_ratings_sparse)
sigma = np.diag(sigma)

# Reconstruct the ratings matrix
U_sigma_Vt = np.dot(np.dot(U, sigma), Vt)

# Add the average ratings back in
uncentered_ratings = U_sigma_Vt + avg_ratings.values.reshape(-1, 1)

# Create a DataFrame of the predicted ratings
calc_pred_ratings_df = pd.DataFrame(uncentered_ratings, 
                                    index=train_df.index, 
                                    columns=train_df.columns)



# Extract the ground truth from the test set
actual_values = test_df.values

# Reindex the predicted ratings to match the test DataFrame
predicted_values = calc_pred_ratings_df.reindex_like(test_df).values

# Create a mask for non-NaN values in the actual test set
mask = ~np.isnan(actual_values)

# Filter out NaN values from both actual and predicted values using the mask
filtered_actual_values = actual_values[mask]
filtered_predicted_values = predicted_values[mask]

# Ensure no NaN values remain in the filtered arrays
filtered_actual_values = filtered_actual_values[~np.isnan(filtered_predicted_values)]
filtered_predicted_values = filtered_predicted_values[~np.isnan(filtered_predicted_values)]

# Calculate RMSE on the test set, considering only non-NaN values
test_rmse = mean_squared_error(filtered_actual_values, filtered_predicted_values, squared=False)
print(f'Test RMSE: {test_rmse}')
test_mae = mean_absolute_error(filtered_actual_values, filtered_predicted_values)
print((f'Test MAE: {test_mae}'))

# Sort the ratings of User 5 from high to low
user_5_ratings = calc_pred_ratings_df.loc['ffff9de9f9ab522578ff9f1b188def1b7375a68f',:].sort_values(ascending=False)

user_rated_tracks = train_df.loc['ffff9de9f9ab522578ff9f1b188def1b7375a68f',:].dropna().index

print(user_5_ratings)

predicted_ratings = calc_pred_ratings_df.loc['ffff9de9f9ab522578ff9f1b188def1b7375a68f'].drop(user_rated_tracks, errors='ignore')

sorted_predicted_ratings = predicted_ratings.sort_values(ascending=False).head(15)

recommendations_df = pd.merge(sorted_predicted_ratings, collab_df[['track_id', 'total_play_count']],
                                  on='track_id', how='left').drop_duplicates()

sorted_recommendations = recommendations_df.sort_values(by=['total_play_count'], ascending=True).head(5)

def recommend_tracks(user, train_df, calc_pred_ratings_df, collab_df):
    #user_ratings = calc_pred_ratings_df.loc[user,:].sort_values(ascending=False)
    # Step 1: Get all tracks that the user has already rated
    user_rated_tracks = train_df.loc[user,:].dropna().index
    # Step 2: Get predicted ratings for tracks the user has not rated
    predicted_ratings = calc_pred_ratings_df.loc[user].drop(user_rated_tracks, errors='ignore')

    # Step 3: Sort these predictions by rating (descending)
    sorted_predicted_ratings = predicted_ratings.sort_values(ascending=False).head(15)
    
    # join predictions with play count
    recommendations_df = pd.merge(sorted_predicted_ratings, collab_df[['track_id', 'total_play_count']],
                                  on='track_id', how='left').drop_duplicates()
    
    sorted_recommendations = recommendations_df.sort_values(by=['total_play_count'], ascending=True).head(5)

    # Step 6: Return the top 25 tracks
    return sorted_recommendations

# Example Usage:
# Assuming 'collab_df' contains user ratings data
random_user = train_df.sample(n=1).iloc[0]['user']
recommendation_collab = recommend_tracks(random_user, train_df, calc_pred_ratings_df, collab_df)

recommendation_collab_with_title = pd.merge(recommendation_collab,track_features_all_df[["title","artist_name","track_id"]],on='track_id', how='left')




########################################################## average base line model ratings #######################################################
##################################################################################################################################################

test_collab = collab_df.copy()
#filtering songs only listened to less than 3 times 
test_collab['play_count'].mean() # this is 3
test_collab = test_collab[(test_collab['play_count'] >= 2)]

test_collab.sort_values(by="play_count", ascending=True).head(10)
test_collab.sort_values(by="total_play_count", ascending=True).head(10)

# only include users that have listened to 10 songs or more 
user_song_counts = test_collab.groupby('user')['track_id'].nunique()
user_song_counts.mean() # this is 1
users_more_than_10_songs = user_song_counts[user_song_counts >= 2].index

# Filter the DataFrame to include only users who have listened to more than 10 songs
test_collab = test_collab[test_collab['user'].isin(users_more_than_10_songs)]

# Show the shape of the new DataFrame to verify
print("New Shape: ", test_collab.shape)

# Display the first few rows to verify the data
test_collab.head()

test_collab

# normalise

def min_max_normalize(series):
    return (series - series.min()) / (series.max() - series.min()) * 4 + 1

# Apply normalization to play_count
test_collab['rating'] = min_max_normalize(test_collab['play_count'])

# Create a user-item rating matrix
user_ratings_df = test_collab.pivot(index='user', columns='track_id', values='rating')

# Flatten the DataFrame into a list of (user, track_id, rating) tuples
ratings = user_ratings_df.stack().reset_index()
ratings.columns = ['user', 'track_id', 'rating']

# Split into train and test sets
train_data, test_data = train_test_split(ratings, test_size=0.2, random_state=42)

# Convert train and test sets back to pivot format
train_df = train_data.pivot(index='user', columns='track_id', values='rating')
test_df = test_data.pivot(index='user', columns='track_id', values='rating')

# Step 1: Calculate the average rating for each user in the training set
avg_ratings = train_df.mean(axis=1)

# Create a DataFrame with the average ratings for all items
predicted_ratings_df = train_df.copy()
predicted_ratings_df[:] = avg_ratings.values.reshape(-1, 1)

# Extract the ground truth from the test set
actual_values = test_df.values

# Reindex the predicted ratings to match the test DataFrame
predicted_values = predicted_ratings_df.reindex_like(test_df).values

# Create a mask for non-NaN values in the actual test set
mask = ~np.isnan(actual_values)

# Filter out NaN values from both actual and predicted values using the mask
filtered_actual_values = actual_values[mask]
filtered_predicted_values = predicted_values[mask]

# Ensure no NaN values remain in the filtered arrays
filtered_actual_values = filtered_actual_values[~np.isnan(filtered_predicted_values)]
filtered_predicted_values = filtered_predicted_values[~np.isnan(filtered_predicted_values)]

# Calculate RMSE on the test set, considering only non-NaN values
test_rmse = mean_squared_error(filtered_actual_values, filtered_predicted_values, squared=False)
print(f'Test RMSE: {test_rmse}')
test_mae = mean_absolute_error(filtered_actual_values, filtered_predicted_values)
print(f'Test MAE: {test_mae}')
