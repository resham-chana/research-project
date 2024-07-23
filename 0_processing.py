# importing relevant libraries
import numpy as np
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import requests
from pathlib import Path

# defining paths
db_track_path = r'C:\Users\resha\data\track_metadata.db'
db_tag_path = r'C:\Users\resha\data\lastfm_tags.db'
triplets_path = r'C:\Users\resha\data\train_triplets.txt'
genre_path = r'C:\Users\resha\data\msd-MAGD-genreAssignment.cls'
image_path = r'C:\Users\resha\data\MSD-I_dataset.tsv'
geography_path = r'C:\Users\resha\data\countries.csv'
#db_track_path = r'C:\Users\corc4\Downloads\track_metadata.db'
#db_tag_path = r'C:\Users\corc4\Downloads\lastfm_tags.db'
#triplets_path = r"C:\Users\corc4\Downloads\train_triplets.txt"
#genre_path = r'C:\Users\corc4\Downloads\msd-MAGD-genreAssignment.cls'
#image_path = r'C:\Users\corc4\Downloads\MSD-I_dataset.tsv'
#geography_path = r'C:\Users\corc4\Downloads\countries.csv'

# read the country data, keep relevant columns and save as a csv
geography_df = pd.read_csv(geography_path)
geography_df = geography_df[["Country","Geography: Map references",
                             "People and Society: Nationality - adjective",
                             "People and Society: Ethnic groups","People and Society: Languages",
                             "People and Society: Religions"]]

# new column names
dict = {'Country': 'country',
        'Geography: Map references': 'continent',
        'People and Society: Nationality - adjective': 'nationality',
        'People and Society: Ethnic groups': 'ethnicity',
        'People and Society: Languages' : 'language',
        'People and Society: Religions': 'religion'}

geography_df.rename(columns=dict,
          inplace=True)

#geography_df.to_csv(r"C:\Users\resha\data\geography_df.csv")  
geography_df.to_csv(r"C:\Users\corc4\data\geography_df.csv")  

# read train_triplet text file and add headers
train_triplets_df = pd.read_table(triplets_path, sep='\t', header=None, names=['user', 'song', 'play_count'])
print(train_triplets_df.head())
# group to find total play count
play_count_grouped_df = train_triplets_df.groupby('song', as_index=False)['play_count'].sum().rename(columns={'play_count': 'total_play_count'})
play_count_grouped_df = play_count_grouped_df.sort_values(by='total_play_count', ascending=False)

# write grouped play count data and triplets to csv
train_triplets_df.to_csv(r"C:\Users\resha\data\train_triplets_df.csv")  
#train_triplets_df.to_csv(r"C:\Users\corc4\data\train_triplets_df.csv")  
play_count_grouped_df.to_csv(r"C:\Users\resha\data\play_count_grouped_df.csv")  
#play_count_grouped_df.to_csv(r"C:\Users\corc4\data\play_count_grouped_df.csv")  

# housekeeping: check column names, unique tracks and nulls
train_triplets_df
train_triplets_df.columns 
# unique tracks
len(pd.unique(train_triplets_df['track_id']))
print(train_triplets_df.isnull().sum().sum())

# Read the image dataset:
images_df = pd.read_csv(image_path, sep='\t')
images_df.head()

# housekeeping: check counts, unique genres and tracks
print(images_df['genre'].value_counts())
images_df['genre'].unique()
images_df['msd_track_id'].nunique()
 
# write to csv
images_df.to_csv(r"C:\Users\resha\data\images_df.csv")  
#images_df.to_csv(r"C:\Users\corc4\data\images_df.csv")  

# Function to download an image from a URL
def download_image(url, path):
    try:
        response = requests.get(url, stream=True, timeout=10)  
        response.raise_for_status()  
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

# iterate through the dataset and download images
for index, row in images_df.iterrows():
    set_name = row['set']
    genre = row['genre']
    url = row['image_url']
    
    # skips the duplicates is already downloaded
    if url in downloaded_urls:
        print(f"Skipping already downloaded URL: {url}")
        continue
    
    # create directory
    dir_path = Path(base_dir) / set_name / genre
    dir_path.mkdir(parents=True, exist_ok=True)
    
    # Define the image path
    image_path = dir_path / f"{row['msd_track_id']}.jpg"
    
    # Download the image
    download_image(url, image_path)
    
    # Add the URL to the set of downloaded URLs
    downloaded_urls.add(url)

# code to covert the track metadata database into a pandas data frame
conn = sqlite3.connect(db_track_path)
c = conn.cursor()
# the table name is 'songs'
TABLENAME = 'songs'
# get data from database and convert to pandas dataframe
q = f"SELECT * FROM {TABLENAME}"
track_metadata_df = pd.read_sql_query(q, conn)
print(track_metadata_df.head()) 
c.close()
conn.close()

# write track metadata to csv
track_metadata_df.to_csv(r"C:\Users\resha\data\track_metadata_df.csv")  
#track_metadata_df.to_csv(r"C:\Users\corc4\data\track_metadata_df.csv")  

# housekeeping: check columns, unique tracks and nulls
track_metadata_df
track_metadata_df.columns 
# unique tracks
len(pd.unique(track_metadata_df['track_id']))
print(track_metadata_df.isnull().sum().sum())

# converting last.fm tag dataset to a pandas datafram
conn_tag = sqlite3.connect(db_tag_path)
# get data of tags and track ID
q_tag = """
SELECT tids.tid, tags.tag
FROM tids
JOIN tid_tag ON tids.ROWID = tid_tag.tid
JOIN tags ON tid_tag.tag = tags.ROWID
"""
lastfm_tags_df = pd.read_sql_query(q_tag, conn_tag)
print(lastfm_tags_df.head())  
conn_tag.close()

# housekeeping: check unique tags, ids
unique_tags_count = lastfm_tags_df['tag'].nunique() #522366
unique_track_count = len(pd.unique(lastfm_tags_df['tid']))
print(lastfm_tags_df.isnull().sum().sum())
# how many tags max does a songs have
lastfm_tags_df.sort_values(by=['tag_number'])

# pivoting tags 
lastfm_tags_df['tag_number'] = lastfm_tags_df.groupby('tid').cumcount() + 1
lastfm_pivot_df = lastfm_tags_df.pivot(index='tid', columns='tag_number', values='tag').reset_index()
lastfm_pivot_df.columns = ['tid'] + [f'tag{i}' for i in range(1, len(lastfm_pivot_df.columns))]

# write csv
lastfm_tags_df.to_csv(r"C:\Users\resha\data\lastfm_tags_df.csv")  
lastfm_pivot_df.to_csv(r"C:\Users\resha\data\lastfm_pivot_df.csv")  
#lastfm_tags_df.to_csv(r"C:\Users\corc4\data\lastfm_tags_df.csv")  
#lastfm_pivot_df.to_csv(r"C:\Users\corc4\data\lastfm_pivot_df.csv") 

# opening genre dataset and coverting to dataframe 
genres_df = pd.read_csv(genre_path,delimiter="\t", header=None)
genres_df.columns = ["track_id","genre"]
print(genres_df)
genres_df["track_id"].nunique

genres_df.to_csv(r"C:\Users\resha\data\genres_df.csv")  
#genres_df.to_csv(r"C:\Users\corc4\data\genres_df.csv")  

lastfm_pivot_df #all tags for each unique song nrow: 505216
lastfm_tags_df # song list with tags that are not unique: 8598630
track_metadata_df # all track metadata from millionsongdataset: 1000000
train_triplets_df # triplets containing information about users and playcounts: 48373586
play_count_grouped_df # play count for each song: 384546
genres_df # genres with 422,714 labels
