import pandas as pd

#https://github.com/d-elicio/Music-Recommender-System-from-scratch/blob/main/Music_Recommender_Project.ipynb

lastfm_diverse_tags_df = pd.read_csv(r"C:\Users\resha\data\lastfm_diverse_tags_df.csv")
lastfm_diverse_pivot_df = pd.read_csv(r"C:\Users\resha\data\lastfm_diverse_pivot_df.csv")
track_metadata_cleaned_df = pd.read_csv(r"C:\Users\resha\data\track_metadata_cleaned_df.csv")  
train_triplets_df = pd.read_csv(r"C:\Users\resha\data\train_triplets_df.csv")
play_count_grouped_df = pd.read_csv(r"C:\Users\resha\data\play_count_grouped_df.csv")
genres_df = pd.read_csv(r"C:\Users\resha\data\genres_df.csv")

track_features_df = pd.merge(track_metadata_cleaned_df, play_count_grouped_df.iloc[:,[1,2]], left_on='song_id', right_on='song').drop('song', axis=1)
track_features_df = pd.merge(track_features_df, genres_df.iloc[:,[1,2]], how='inner', on='track_id')
track_features_df = pd.merge(track_features_df, train_triplets_df, how='inner', on='track_id')
track_features_df.dropna(inplace= True)

tf_df = track_features_df.copy()

user_matrix = tf_df.pivot_table(index='user_id', 
                                    columns='title', values='play_count', fill_value=0)

sparsity = 1.0 - (((user_matrix == 0).sum().sum()) / float(user_matrix.size))

