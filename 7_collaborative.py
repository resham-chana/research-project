# import relevant libraries
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import svds
import scipy.sparse as sp

train_triplets_df = pd.read_csv(r"C:\Users\resha\data\train_triplets_df.csv")
track_features_all_df = pd.read_csv(r"C:\Users\resha\data\track_features_all_df.csv")
content_df = pd.read_csv(r"C:\Users\resha\data\content_df.csv")

# create new dataframe including triplets
collab_df = pd.merge(content_df, train_triplets_df, left_on="song_id", right_on="song", how="left")
collab_df = collab_df.dropna(subset=['user'])
collab_df = collab_df.drop(["Unnamed: 0.2", "Unnamed: 0.1","Unnamed: 0_x",
                            'Unnamed: 0_y',"song","title","song_id","release",
                            "artist_id",'artist_mbid',
                            'artist_name', 'duration', 'artist_familiarity', 'artist_hotttnesss','year','genre', 
                            'log_total_play_count', 'country',
                            'gender', 'language', 'religion', 'continent', 'cleaned_tag', 'decade'], axis=1)
collab_df.sort_values(by="play_count")

########### outlier removal with IQR experiment ###########

# code adapted from [https://www.geeksforgeeks.org/detect-and-remove-the-outliers-using-python/]
# get play counts
play_count_values = collab_df['play_count'].value_counts()
collab_df["play_count"].mean()
collab_df["total_play_count"].max()

# copy for experiment
test = collab_df.copy()

# upper and lower limits
Q1 = test['total_play_count'].quantile(0.25)
Q3 = test['total_play_count'].quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5*IQR
upper = Q3 + 1.5*IQR

# iltering songs only listened to less than 3 times 
collab_df['play_count'].mean() 
collab_df = collab_df[(collab_df['play_count'] >= 3)]

# only include users that have listened to 2 songs or more 
user_song_counts = collab_df.groupby('user')['track_id'].nunique()
user_song_counts.mean() # this is 1
users_more_than_2_songs = user_song_counts[user_song_counts >= 2].index
collab_df = collab_df[collab_df['user'].isin(users_more_than_2_songs)]

# normalise play count into a rating from 1 to 5
def min_max_normalise(series):
    '''
    function to return a normalised rating from 1 to 5
    '''
    return (series - series.min()) / (series.max() - series.min()) * 4 + 1

collab_df['rating'] = min_max_normalise(collab_df['play_count'])

# check the distribution
collab_df['rating'].plot(kind='hist', bins=1000, figsize=(10, 6))
plt.show()

########## collaborative filtering system with matrix factorisation (SVD) ##########

# code adapted from [https://campus.datacamp.com/courses/building-recommendation-engines-in-python/matrix-factorization-and-validating-your-predictions?ex=9]

user_ratings_df = collab_df.pivot(index='user',columns='track_id',values='rating')

# count occupied cells
sparsity_count = user_ratings_df.isnull().values.sum()

# count all cells
full_count = user_ratings_df.size

# calculate the sparsity
sparsity = sparsity_count / full_count
print(sparsity)

ratings = user_ratings_df.stack().reset_index()
ratings.columns = ['user', 'track_id', 'rating']

# split into train and test sets
train_data, test_data = train_test_split(ratings, test_size=0.2, random_state=42)

# convert train and test sets back to pivot format
train_df = train_data.pivot(index='user', columns='track_id', values='rating')
test_df = test_data.pivot(index='user', columns='track_id', values='rating')

# average rating for each user in the training set
avg_ratings = train_df.mean(axis=1)
# centre around 0
user_ratings_centered_train_df = train_df.sub(avg_ratings, axis=0)
# fill NaNs with 0
user_ratings_centered_train_df.fillna(0, inplace=True)
# convert to sparse matrix
user_ratings_sparse = sp.csr_matrix(user_ratings_centered_train_df.values)

# perform SVD matrix factorisation 
U, sigma, Vt = svds(user_ratings_sparse)
sigma = np.diag(sigma)

# reconstruct ratings matrix
U_sigma_Vt = np.dot(np.dot(U, sigma), Vt)

# add average ratings back
uncentered_ratings = U_sigma_Vt + avg_ratings.values.reshape(-1, 1)

# predict ratings with uncentered_ratings df
calc_pred_ratings_df = pd.DataFrame(uncentered_ratings, 
                                    index=train_df.index, 
                                    columns=train_df.columns)


########### evaluate SVD ###########

# get actual values from test set
actual_values = test_df.values

# reindex the predicted ratings to match the test set
predicted_values = calc_pred_ratings_df.reindex_like(test_df).values

# create a lit of bools for non NaN values in the test set
mask = ~np.isnan(actual_values)

# filter out NaN values from both actual and predicted values
filtered_actual_values = actual_values[mask]
filtered_predicted_values = predicted_values[mask]

# check filtered arrays for NaNs
filtered_actual_values = filtered_actual_values[~np.isnan(filtered_predicted_values)]
filtered_predicted_values = filtered_predicted_values[~np.isnan(filtered_predicted_values)]

# RMSE on the test set
test_rmse_svd = mean_squared_error(filtered_actual_values, filtered_predicted_values, squared=False)
print(test_rmse_svd)

# MAE on the test set
test_mae_svd = mean_absolute_error(filtered_actual_values, filtered_predicted_values)
print(test_mae_svd)

# sort the ratings of user ffff9de9f9ab522578ff9f1b188def1b7375a68f from high to low
user_5_ratings = calc_pred_ratings_df.loc['ffff9de9f9ab522578ff9f1b188def1b7375a68f',:].sort_values(ascending=False)
user_rated_tracks = train_df.loc['ffff9de9f9ab522578ff9f1b188def1b7375a68f',:].dropna().index

print(user_5_ratings)

# predict ratings for this user 
predicted_ratings = calc_pred_ratings_df.loc['ffff9de9f9ab522578ff9f1b188def1b7375a68f'].drop(user_rated_tracks, errors='ignore')

# sort
sorted_predicted_ratings = predicted_ratings.sort_values(ascending=False).head(15)

# find the play count
recommendations_df = pd.merge(sorted_predicted_ratings, collab_df[['track_id', 'total_play_count']],
                                  on='track_id', how='left').drop_duplicates()

# get recs with lower play count
sorted_recommendations = recommendations_df.sort_values(by=['total_play_count'], ascending=True).head(5)

def recommend_tracks(user, train_df, calc_pred_ratings_df, collab_df):
    '''
    function to recommend tracks with lower play count but still with the highest predicted ratings
    '''
    # tracks that the user has already rated
    user_rated_tracks = train_df.loc[user,:].dropna().index
    # predicted ratings for tracks the user has not rated
    predicted_ratings = calc_pred_ratings_df.loc[user].drop(user_rated_tracks, errors='ignore')
    # sort ratings
    sorted_predicted_ratings = predicted_ratings.sort_values(ascending=False).head(15)
    # join predictions with play count
    recommendations_df = pd.merge(sorted_predicted_ratings, collab_df[['track_id', 'total_play_count']],
                                  on='track_id', how='left').drop_duplicates()
    # take the top 5 of the lowest played songs
    sorted_recommendations = recommendations_df.sort_values(by=['total_play_count'], ascending=True).head(5)
    return sorted_recommendations

# take a random user and apply the function
random_user = train_df.sample(n=1).index[0]
random_user_rated_tracks = train_df.loc[random_user,:].dropna()
random_user_rated_tracks_with_title = pd.merge(random_user_rated_tracks,track_features_all_df[["title","artist_name","track_id"]],on='track_id', how='left')
recommendation_collab = recommend_tracks(random_user, train_df, calc_pred_ratings_df, collab_df)

recommendation_collab_with_title = pd.merge(recommendation_collab,track_features_all_df[["title","artist_name","track_id"]],on='track_id', how='left')

########### average base line model ratings ###########

# filtering the test df
test_collab = collab_df.copy()
test_collab['play_count'].mean() 
test_collab = test_collab[(test_collab['play_count'] >= 2)]

# only include users that have listened to 2 songs or more 
user_song_counts = test_collab.groupby('user')['track_id'].nunique()
user_song_counts.mean() 
users_more_than_10_songs = user_song_counts[user_song_counts >= 2].index
test_collab = test_collab[test_collab['user'].isin(users_more_than_10_songs)]

# Apply normalization to play_count
test_collab['rating'] = min_max_normalise(test_collab['play_count'])

# user ratings in a pivoted df
user_ratings_df = test_collab.pivot(index='user', columns='track_id', values='rating')

# flatten the df into a list of tuples
ratings = user_ratings_df.stack().reset_index()
ratings.columns = ['user', 'track_id', 'rating']

# split into train and test sets
train_data, test_data = train_test_split(ratings, test_size=0.2, random_state=42)

# convert to pivot format
train_df = train_data.pivot(index='user', columns='track_id', values='rating')
test_df = test_data.pivot(index='user', columns='track_id', values='rating')

# calculate the average rating for each user in the training set
avg_ratings = train_df.mean(axis=1)

# new df with the average ratings for all items
predicted_ratings_df = train_df.copy()
predicted_ratings_df[:] = avg_ratings.values.reshape(-1, 1)

# extract the ground truth from the test set
actual_values = test_df.values

# reindex predicted ratings to match the test df
predicted_values = predicted_ratings_df.reindex_like(test_df).values

# mask of bools to filter NaNs
mask = ~np.isnan(actual_values)
filtered_actual_values = actual_values[mask]
filtered_predicted_values = predicted_values[mask]

# check for NaNs
filtered_actual_values = filtered_actual_values[~np.isnan(filtered_predicted_values)]
filtered_predicted_values = filtered_predicted_values[~np.isnan(filtered_predicted_values)]

# RMSE 
test_rmse_baseline = mean_squared_error(filtered_actual_values, filtered_predicted_values, squared=False)
print(test_rmse_baseline)
# MAE
test_mae_baseline = mean_absolute_error(filtered_actual_values, filtered_predicted_values)
print(test_mae_baseline)
