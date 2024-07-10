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
import requests
from pathlib import Path
from thefuzz import fuzz

db_track_path = r'C:\Users\resha\data\track_metadata.db'
db_tag_path = r'C:\Users\resha\data\lastfm_tags.db'
triplets_path = r'C:\Users\resha\data\train_triplets.txt'
genre_path = r'C:\Users\resha\data\msd-MAGD-genreAssignment.cls'
image_path = r'C:\Users\resha\data\MSD-I_dataset.tsv'
#db_track_path = r'C:\Users\corc4\Downloads\track_metadata.db'
#db_tag_path = r'C:\Users\corc4\Downloads\lastfm_tags.db'
#triplets_path = r"C:\Users\corc4\Downloads\train_triplets.txt"
#genre_path = r'C:\Users\corc4\Downloads\msd-MAGD-genreAssignment.cls'
#image_path = r'C:\Users\corc4\Downloads\MSD-I_dataset.tsv'

# https://www.ifs.tuwien.ac.at/mir/msd/


# Read the tab-delimited text file and add headers
train_triplets_df = pd.read_table(triplets_path, sep='\t', header=None, names=['user', 'song', 'play_count'])
print(train_triplets_df.head())
play_count_grouped_df = train_triplets_df.groupby('song', as_index=False)['play_count'].sum().rename(columns={'play_count': 'total_play_count'})
play_count_grouped_df = play_count_grouped_df.sort_values(by='total_play_count', ascending=False)

# housekeeping
train_triplets_df
train_triplets_df.columns 
# unique tracks
len(pd.unique(train_triplets_df['track_id']))
print(train_triplets_df.isnull().sum().sum())


# Read the image dataset:
images_df = pd.read_csv(image_path, sep='\t')
images_df.head()

# housekeeping 
print(images_df['genre'].value_counts())
images_df['genre'].unique()


# Function to download an image from a URL
def download_image(url, path):
    try:
        response = requests.get(url, stream=True, timeout=10)  # timeout added
        response.raise_for_status()  # Raises an HTTPError 
        with open(path, 'wb') as file:
            for chunk in response.iter_content(1024): 
                file.write(chunk)
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred while downloading {url}: {http_err}")
    except requests.exceptions.ConnectionError as conn_err:
        print(f"Connection error occurred while downloading {url}: {conn_err}")
    except requests.exceptions.Timeout as timeout_err:
        print(f"Timeout error occurred while downloading {url}: {timeout_err}")
    except requests.exceptions.RequestException as req_err:
        print(f"An error occurred while downloading {url}: {req_err}")
    except Exception as e:
        print(f"An unexpected error occurred while downloading {url}: {e}")

# Base directory for the images 
#base_dir = 'C:\Users\resha\images'
#base_dir = r'C:\Users\corc4\Downloads\images'
# Set to keep track of downloaded URLs
downloaded_urls = set()

# Iterate through the dataset and download images
for index, row in images_df.iterrows():
    set_name = row['set']
    genre = row['genre']
    url = row['image_url']
    
    # Check if the URL has already been downloaded - skips the duplicates
    if url in downloaded_urls:
        print(f"Skipping already downloaded URL: {url}")
        continue
    
    # Create the directory if it does not exist
    dir_path = Path(base_dir) / set_name / genre
    dir_path.mkdir(parents=True, exist_ok=True)
    
    # Define the image path
    image_path = dir_path / f"{row['msd_track_id']}.jpg"
    
    # Download the image
    download_image(url, image_path)
    
    # Add the URL to the set of downloaded URLs
    downloaded_urls.add(url)

print("Image download and organisation complete.")

# code to covert the track metadata database into a pandas data frame
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

# housekeeping
track_metadata_df
track_metadata_df.columns 
# unique tracks
len(pd.unique(track_metadata_df['track_id']))
print(track_metadata_df.isnull().sum().sum())

# converting last.fm tag dataset to a pandas datafram
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

# housekeeping
unique_tags_count = lastfm_tags_df['tag'].nunique() #522366
unique_track_count = len(pd.unique(lastfm_tags_df['tid']))
print(lastfm_tags_df.isnull().sum().sum())
# how many tags max does a songs have
lastfm_tags_df.sort_values(by=['tag_number'])

# pivoting tags 
lastfm_tags_df['tag_number'] = lastfm_tags_df.groupby('tid').cumcount() + 1
lastfm_df = lastfm_tags_df.pivot(index='tid', columns='tag_number', values='tag').reset_index()
lastfm_df.columns = ['tid'] + [f'tag{i}' for i in range(1, len(lastfm_df.columns))]

# extracting male/female artists and geographic 
lastfm_df 

# opening genre dataset and coverting to dataframe 

genres_df1 = pd.read_csv(genre_path,delimiter="\t", header=None)
genres_df1.columns = ["track_id","genre"]
print(genres_df1)

lastfm_df #all tags for each unique song nrow: 505216
lastfm_tags_df # song list with tags that are not unique: 8598630
track_metadata_df # all track metadata from millionsongdataset: 1000000
train_triplets_df # triplets containing information about users and playcounts: 48373586
play_count_grouped_df # play count for each song: 384546
genres_df1 # genres with 422,714 labels

# joining dataset 
track_df = pd.merge(track_metadata_df, play_count_grouped_df, left_on='song_id', right_on='song').drop('song', axis=1)
track_df = pd.merge(track_df, genres_df1, how='inner', on='track_id')
track_df = pd.merge(track_df, lastfm_df, how='left', left_on='track_id', right_on='tid').drop('tid', axis=1)

track_df # nrow: 385256 columns (with genre this goes down to 195002): ['track_id', 'title', 'song_id', 'release', 'artist_id', 
         #'artist_mbid', 'artist_name', 'duration', 'artist_familiarity', 'artist_hotttnesss', 'year', 'total_play_count', 'tag1', "genre"]

track_df2 = pd.merge(track_metadata_df, play_count_grouped_df, left_on='song_id', right_on='song').drop('song', axis=1)
track_df2 = pd.merge(track_df, images_df, how='inner',  left_on='track_id', right_on='msd_track_id')
track_df2 = pd.merge(track_df, lastfm_df, how='left', left_on='track_id', right_on='tid').drop('tid', axis=1)

track_df2 # nrow: 385256 columns (with genre this goes down to 195002): ['track_id', 'title', 'song_id', 'release', 'artist_id', 
         #'artist_mbid', 'artist_name', 'duration', 'artist_familiarity', 'artist_hotttnesss', 'year', 'total_play_count', 'tag1', "genre"]

track_df.to_csv(r"C:\Users\resha\research-project\data\track_df_genre1.csv")  
track_df2.to_csv(r"C:\Users\resha\research-project\data\track_df_genre2.csv")  

# housekeeping 

print(track_df.isnull().sum().sum())
print(track_df2.isnull().sum().sum())


# test NANs and rows etc
track_df.columns

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

#"male", "female",""

