import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

track_metadata_df = pd.read_csv(r"C:\Users\corc4\data\track_metadata_df.csv")

track_metadata_cleaned_df = track_metadata_df.drop(columns=['Unnamed: 0','shs_work','shs_perf','track_7digitalid','artist_id','artist_mbid',])

track_metadata_cleaned_df.columns 

#'track_id', 'title', 'song_id', 'release', 'artist_id',
#       'artist_mbid', 'artist_name', 'duration', 'artist_familiarity',
#       'artist_hotttnesss', 'year', 'track_7digitalid', 'shs_perf',
#       'shs_work'

track_metadata_cleaned_df = track_metadata_df[track_metadata_df['year'] != 0]


# Get the value counts of the 'year' column
year_counts = track_metadata_cleaned_df['year'].value_counts().sort_index()

# Set the plot style
plt.style.use("dark_background")
# Create the figure and axis
fig, ax = plt.subplots(figsize=(12, 8))
# Plot the data
year_counts.plot(kind="bar", color="#4CAF50", ax=ax)
# Set labels and title
ax.set_xlabel('Year')
ax.set_ylabel('Count')
ax.set_title('Track Count by Year')
plt.xticks(fontsize=8)
# Show the plot
#plt.savefig(r"C:\Users\resha\plots\year_distribution.png")
#plt.savefig(r"C:\Users\corc4\plots\year_distribution.png")
plt.savefig("year_distribution.png")
plt.show()
