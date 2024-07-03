import os
import sys
import glob
import time
import datetime
import numpy as np
import sqlite3
import csv
import pandas as pd
import matplotlib.pyplot as plt
 

# Connect to the SQLite database

db_track_path = r'C:\Users\resha\Documents\RESH\Data Science\Research Project\data\millionsongsubset\track_metadata.db'
db_tag_path = r'C:\Users\resha\Documents\RESH\Data Science\Research Project\data\lastfm_tags.db'
triplets_path = r"C:\Users\resha\Documents\RESH\Data Science\Research Project\data\train_triplets.txt"


# Read the tab-delimited text file and add headers
train_triplets_df = pd.read_table(triplets_path, sep='\t', header=None, names=['user', 'song', 'play_count'])
print(train_triplets_df.head())
play_count_grouped_df = train_triplets_df.groupby('song', as_index=False)['play_count'].sum().rename(columns={'play_count': 'total_play_count'})
play_count_grouped_df = play_count_grouped_df.sort_values(by='total_play_count', ascending=False)

# connect to the SQLite database
conn = sqlite3.connect(db_track_path)
# from that connection, get a cursor to do queries
c = conn.cursor()
# the table name is 'songs'
TABLENAME = 'songs'
# Fetch data from SQLite database and convert to Pandas DataFrame
q = f"SELECT * FROM {TABLENAME}"
track_metadata_df = pd.read_sql_query(q, conn)
# Display the DataFrame
print('*************** TABLE CONTENT AS PANDAS DATAFRAME ***************************')
print(track_metadata_df.head())  # Display the first few rows
# close the cursor and the connection
c.close()
conn.close()

# Connect to the lastfm_tags SQLite database
conn_tag = sqlite3.connect(db_tag_path)

# Fetch data with tids in one column and tags in another column
q_tag = """
SELECT tids.tid, tags.tag
FROM tids
JOIN tid_tag ON tids.ROWID = tid_tag.tid
JOIN tags ON tid_tag.tag = tags.ROWID
"""
lastfm_tags_df = pd.read_sql_query(q_tag, conn_tag)

print('*************** LASTFM TAGS AS PANDAS DATAFRAME ***************************')
print(lastfm_tags_df.head())  # Display the first few rows

conn_tag.close()

unique_tags_count = lastfm_tags_df['tag'].nunique() #522366

lastfm_tags_df['tag_number'] = lastfm_tags_df.groupby('tid').cumcount() + 1
lastfm_df = lastfm_tags_df.pivot(index='tid', columns='tag_number', values='tag').reset_index()
lastfm_df.columns = ['tid'] + [f'tag{i}' for i in range(1, len(lastfm_df.columns))]


lastfm_df #all tags for each unique song nrow: 505216
lastfm_tags_df # song list with tags that are not unique: 8598630
track_metadata_df # all track metadata from millionsongdataset: 1000000
train_triplets_df # triplets containing information about users and playcounts: 48373586
play_count_grouped_df # play count for each song: 384546

# joining dataset 
track_df = pd.merge(track_metadata_df, play_count_grouped_df, left_on='song_id', right_on='song').drop('song', axis=1)
track_df = pd.merge(track_df, lastfm_df, how='left', left_on='track_id', right_on='tid').drop('tid', axis=1)

track_df # nrow: 385256 columns: ['track_id', 'title', 'song_id', 'release', 'artist_id', 
         #'artist_mbid', 'artist_name', 'duration', 'artist_familiarity', 'artist_hotttnesss', 'year', 'total_play_count', 'tag1'


#filtered = track_df[track_df["year"]>0]["year"]
#plays = track_df[track_df["total_play_count"]>0]

fig, ax = plt.subplots(figsize=(10, 6))

# Scatter plot
ax.scatter(track_df[track_df["year"]>0]["year"], track_df[track_df["year"]>0]["total_play_count"], marker='o', color='b', alpha=0.7)

# Add labels and title
ax.set_title('Total Play Count vs Year')
ax.set_xlabel('Year')
ax.set_ylabel('Total Play Count')

# Display grid for better visualization
ax.grid(True)

# Show plot
plt.tight_layout()
plt.show()

##################################### TAG EXPLORATION ##################################################
import seaborn as sns

tag_counts = lastfm_tags_df["tag"].value_counts()

# Convert to DataFrame for better handling in visualization
tag_counts_df = tag_counts.reset_index()
tag_counts_df.columns = ['tag', 'count']

# Set the figure size
plt.figure(figsize=(10, 8))

# Create a bar plot
sns.barplot(x='count', y='tag', data=tag_counts_df.head(100))  # Displaying top 20 tags for readability

# Set plot title and labels
plt.title('Top 100 Tags by Count')
plt.xlabel('Count')
plt.ylabel('Tag')

plt.yticks(fontsize=7)

# Show plot
plt.show()

########################## clustering tags ##################################

import gensim.downloader as api
from sklearn.metrics.pairwise import cosine_similarity


glove_model = api.load("glove-wiki-gigaword-300")

glove_model["beautiful"]

lastfm_df

def get_tag_vector(tag, model):
    words = tag.split()
    vectors = [model[word] for word in words if word in model]
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        return None
    
tags = lastfm_df.iloc[:, 1:].stack().unique()  # Get unique tags, excluding NaNs
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

lastfm_df['tags'] = lastfm_df.iloc[:, 1:].apply(lambda row: row.dropna().tolist(), axis=1)

lastfm_df['song_vector'] = lastfm_df['tags'].apply(lambda tags: compute_song_vector(tags, tag_vectors))

song_vectors = np.stack(lastfm_df['song_vector'].values)

similarity_matrix = cosine_similarity(song_vectors)

similarity_df = pd.DataFrame(similarity_matrix, index=lastfm_df['tid'], columns=lastfm_df['tid'])
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



