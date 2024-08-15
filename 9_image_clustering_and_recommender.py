# import relevant libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from colorthief import ColorThief
import seaborn as sns
from sklearn.metrics.pairwise import euclidean_distances

# code adapted from [https://towardsdatascience.com/how-to-cluster-images-based-on-visual-similarity-cd6e7209fe34]

# set path to the dataset location
path = r"C:/Users/resha/images/train"

# change the working directory to the path where the images are located
os.chdir(path)

# hold the images here
albums = []

# create a ScandirIterator aliased as files
for root, dirs, files in os.walk(path):
    for file in files:
        if file.endswith('.jpg'):
            albums.append((os.path.join(root, file), os.path.basename(root)))

# extract colour with colourthief
def extract_main_colour(filepath):
    """Function to extract the main colour from an image"""
    color_thief = ColorThief(filepath)
    dominant_colour = color_thief.get_color(quality=1)  
    return dominant_colour

# store dominant colours and labels
colour_data = {}
labels = {}

# loop through each image in the dataset
for file, label in albums:
    main_colour = extract_main_colour(file)
    colour_data[file] = main_colour
    labels[file] = label

# convert to numpy array
filenames = np.array(list(colour_data.keys()))
colours = np.array(list(colour_data.values()))

# convert colours to float for clustering
colours = colours.astype(float)

colour_df = pd.DataFrame(colours, columns=['R', 'G', 'B'])
colour_df['Label'] = [labels[file] for file in filenames]
colour_df.value_counts(subset="Label")

print(f"Sample from labels dictionary: {list(labels.items())[:5]}")
print(f"Sample filenames: {filenames[:5]}")


# 3D plot 
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(
    colour_df['R'], colour_df['G'], colour_df['B'],
    c=colour_df[['R', 'G', 'B']].values / 255,  
    s=10, 
)
ax.set_xlabel('Red Component')
ax.set_ylabel('Green Component')
ax.set_zlabel('Blue Component')
ax.set_title('Colours by Genre (3D: Red, Green, Blue)')
plt.show()

# compare jazz and metal
labels_to_plot = ["Jazz", "Metal"]
fig = plt.figure(figsize=(15, 15))
num_labels = len(labels_to_plot)
rows = (num_labels + 1) // 2  

# loop through each label to create a subplot
for i, label in enumerate(labels_to_plot):
    ax = fig.add_subplot(rows, 2, i + 1, projection='3d')
    subset = colour_df[colour_df['Label'] == label]
    sc = ax.scatter(
        subset['R'], subset['G'], subset['B'],
        c=subset[['R', 'G', 'B']].values / 255,  
        s=30,
        alpha=0.8 
    )
    ax.set_xlabel('Red Component')
    ax.set_ylabel('Green Component')
    ax.set_zlabel('Blue Component')
    ax.set_title(f'Genre: {label}', fontsize=18) 

plt.tight_layout()
plt.show()


track_titles = [os.path.splitext(os.path.basename(file))[0] for file, _ in albums]
colour_df['track_id'] = track_titles

def recommend_tracks_colour(input_colour, genre, df):
    """
    Function to recommend songs based on the colour of the album artwork
    """
    # filter the dataframe by genre
    imgage_genre_df = df[df['Label'] == genre]
    # calculate the euclidean distances between the input color and the colors in the dataset
    input_colour_array = np.array(input_colour).reshape(1, -1)
    genre_colours = imgage_genre_df[['R', 'G', 'B']].values
    distances = euclidean_distances(input_colour_array, genre_colours).flatten()
    # add distances to the genre dataframe
    imgage_genre_df = imgage_genre_df.copy()
    imgage_genre_df['distance'] = distances
    # sort by distance 
    close_songs = imgage_genre_df.sort_values('distance').head(10)

    return close_songs[['track_id', 'R', 'G', 'B', 'distance']]

input_colour = (120, 150, 200)  
genre = "Jazz"  

# get the top 25 recommended songs
recommended_tracks_from_colour = recommend_tracks_colour(input_colour, genre, colour_df)

print(recommended_tracks_from_colour)