import pandas as pd

lastfm_diverse_tags_df = pd.read_csv(r"C:\Users\resha\data\lastfm_diverse_tags_df.csv")
lastfm_diverse_pivot_df = pd.read_csv(r"C:\Users\corc4\data\lastfm_diverse_pivot_df.csv")
track_metadata_df = pd.read_csv(r"C:\Users\corc4\data\track_metadata_df.csv")
train_triplets_df = pd.read_csv(r"C:\Users\corc4\data\train_triplets_df.csv")
play_count_grouped_df = pd.read_csv(r"C:\Users\corc4\data\play_count_grouped_df.csv")
genres_df = pd.read_csv(r"C:\Users\corc4\data\genres_df.csv")


track_metadata_df.columns
track_df = pd.merge(track_metadata_df, play_count_grouped_df, left_on='song_id', right_on='song').drop('song', axis=1)
track_df = pd.merge(track_df, genres_df, how='inner', on='track_id')
track_df = pd.merge(track_df, lastfm_diverse_pivot_df, how='left', left_on='track_id', right_on='tid').drop('tid', axis=1)
