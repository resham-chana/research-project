import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from colorthief import ColorThief
import seaborn as sns
from sklearn.metrics.pairwise import euclidean_distances

# Set the path to the dataset location
path = r"C:/Users/resha/images/train"

# Change the working directory to the path where the images are located
os.chdir(path)

# This list holds all the image filenames and their respective labels
albums = []

# Create a ScandirIterator aliased as files
for root, dirs, files in os.walk(path):
    for file in files:
        if file.endswith('.jpg'):
            # Save the full path to the file and the class label (folder name)
            albums.append((os.path.join(root, file), os.path.basename(root)))

def extract_main_colour(filepath):
    """Function to extract the main colour from an image"""
    color_thief = ColorThief(filepath)
    dominant_colour = color_thief.get_color(quality=1)  
    return dominant_colour

# Store the dominant colours and labels
colour_data = {}
labels = {}

# Loop through each image in the dataset
for file, label in albums:
    # Extract the main colour
    main_colour = extract_main_colour(file)
    
    # Store the colour and label
    colour_data[file] = main_colour
    labels[file] = label


# Convert to numpy array
filenames = np.array(list(colour_data.keys()))
colours = np.array(list(colour_data.values()))

# Convert colours to float for clustering
colours = colours.astype(float)

colour_df = pd.DataFrame(colours, columns=['R', 'G', 'B'])
colour_df['Label'] = [labels[file] for file in filenames]
colour_df.value_counts(subset="Label")
#colour_df['Cluster'] = kmeans.labels_

print(f"Sample from labels dictionary: {list(labels.items())[:5]}")
print(f"Sample filenames: {filenames[:5]}")


# Plot 3D Scatter Plot (Red, Green, Blue)
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(
    colour_df['R'], colour_df['G'], colour_df['B'],
    c=colour_df[['R', 'G', 'B']].values / 255,  # Normalize RGB values to [0, 1]
    s=10,  # Size of points
)

# Add labels and title
ax.set_xlabel('Red Component')
ax.set_ylabel('Green Component')
ax.set_zlabel('Blue Component')
ax.set_title('Colours by Genre (3D: Red, Green, Blue)')

plt.show()

# Define the labels you want to plot
labels_to_plot = ["Jazz", "Metal"]

# Set up the plot
fig = plt.figure(figsize=(15, 15))
num_labels = len(labels_to_plot)
rows = (num_labels + 1) // 2  # Number of rows for subplots

# Loop through each label to create a subplot
for i, label in enumerate(labels_to_plot):
    ax = fig.add_subplot(rows, 2, i + 1, projection='3d')
    
    # Filter data for the current label
    subset = colour_df[colour_df['Label'] == label]
    
    # Create 3D scatter plot
    sc = ax.scatter(
        subset['R'], subset['G'], subset['B'],
        c=subset[['R', 'G', 'B']].values / 255,  # Normalize RGB values to [0, 1]
        s=30,
        alpha=0.8  # Size of points
    )
    
    # Add labels and title
    ax.set_xlabel('Red Component')
    ax.set_ylabel('Green Component')
    ax.set_zlabel('Blue Component')
    ax.set_title(f'Genre: {label}', fontsize=18)  # Increase title font size

# Adjust layout to prevent overlap
plt.tight_layout()
plt.show()





track_titles = [os.path.splitext(os.path.basename(file))[0] for file, _ in albums]
colour_df['track_id'] = track_titles

def recommend_tracks_colour(input_colour, genre, df):
    """
    Recommend songs based on color similarity within a specific genre.
    
    Parameters:
        input_colour (tuple): RGB values of the input color (e.g., (255, 0, 0) for red).
        genre (str): The genre to filter songs by (e.g., "Jazz").
        df (pd.DataFrame): DataFrame containing 'R', 'G', 'B', 'Label', and 'Song' columns.
        top_n (int): Number of top recommendations to return.
    
    Returns:
        pd.DataFrame: DataFrame containing the recommended songs and their corresponding colors.
    """
    # Filter the dataframe by genre
    imgage_genre_df = df[df['Label'] == genre]
    
    # Compute Euclidean distances between the input color and the colors in the dataset
    input_colour_array = np.array(input_colour).reshape(1, -1)
    genre_colours = imgage_genre_df[['R', 'G', 'B']].values
    distances = euclidean_distances(input_colour_array, genre_colours).flatten()
    
    # Add the distances to the genre dataframe
    imgage_genre_df = imgage_genre_df.copy()
    imgage_genre_df['distance'] = distances
    
    # Sort by distance to find the closest matches
    close_songs = imgage_genre_df.sort_values('distance').head(10)
    
    # Return the recommended songs and their respective colors
    return close_songs[['track_id', 'R', 'G', 'B', 'distance']]

# Example usage:
input_colour = (120, 150, 200)  # Example input color (light blue-ish)
genre = "Jazz"  # Example genre

# Get the top 25 recommended songs
recommended_tracks_from_colour = recommend_tracks_colour(input_colour, genre, colour_df)

print(recommended_tracks_from_colour)