import pandas as pd
# Seaborn visualization library
import seaborn as sns
# Create the default pairplot
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import country_converter as coco

#lastfm_diverse_tags_df = pd.read_csv(r"C:\Users\corc4\data\lastfm_diverse_tags_df.csv")
#lastfm_diverse_pivot_df = pd.read_csv(r"C:\Users\corc4\data\lastfm_diverse_pivot_df.csv")
#track_metadata_cleaned_df = pd.read_csv(r"C:\Users\corc4\data\track_metadata_cleaned_df.csv")  
#train_triplets_df = pd.read_csv(r"C:\Users\corc4\data\train_triplets_df.csv")
#play_count_grouped_df = pd.read_csv(r"C:\Users\corc4\data\play_count_grouped_df.csv")
#genres_df = pd.read_csv(r"C:\Users\corc4\data\genres_df.csv")
lastfm_diverse_tags_df = pd.read_csv(r"C:\Users\resha\data\lastfm_diverse_tags_df.csv")
lastfm_diverse_pivot_df = pd.read_csv(r"C:\Users\resha\data\lastfm_diverse_pivot_df.csv")
track_metadata_df = pd.read_csv(r"C:\Users\resha\data\track_metadata_df.csv")
train_triplets_df = pd.read_csv(r"C:\Users\resha\data\train_triplets_df.csv")
play_count_grouped_df = pd.read_csv(r"C:\Users\resha\data\play_count_grouped_df.csv")
genres_df = pd.read_csv(r"C:\Users\resha\data\genres_df.csv")
#images_df = pd.read_csv(r'C:\Users\corc4\Downloads\MSD-I_dataset.tsv', sep='\t')
#images_df.columns
#images_df = images_df.iloc[:,[0,1,2]]


track_features_df = pd.merge(track_metadata_df, play_count_grouped_df.iloc[:,[1,2]], left_on='song_id', right_on='song', how ="left").drop('song', axis=1)
track_features_df = pd.merge(track_features_df, genres_df.iloc[:,[1,2]], how='left', on='track_id')
#track_features_df.dropna(inplace= True)
# Playcount and genre and year 
# Year and popularity
track_features_df['log_total_play_count'] = np.log10(track_features_df['total_play_count'])

# Drop the non-transformed columns
#track_features_plot_df = track_features_plot_df.drop('total_play_count',axis=1)


selected_vars = ['duration','year', 'artist_familiarity', 'artist_hotttnesss'
                 ,'log_total_play_count']
plt.style.use("dark_background")
sns.pairplot(track_features_df, vars=selected_vars, hue = "genre",  palette="husl"
             ,kind = "kde",plot_kws={'alpha':0.5})
plt.savefig("pairwiseplot.png")
plt.show()


# gender, race etc

track_features_gender_df = pd.merge(track_features_df, lastfm_diverse_pivot_df[['tid', 'gender']], 
                                  how='left', left_on='track_id', right_on='tid').drop('tid', axis=1)


#'track_id', 'title', 'song_id', 
#'release', 'artist_name', 'duration',
#'artist_familiarity', 'artist_hotttnesss', 
#'year', 'total_play_count','genre', "country","language","gender", 


plt.style.use("dark_background")
# Create the figure and axis
fig, ax = plt.subplots(figsize=(12, 8))
# Create the boxplot
sns.boxplot(x='gender', y='log_total_play_count', data=track_features_gender_df,
             palette='viridis', ax=ax)
# Set labels and title
ax.set_xlabel('Gender', fontsize=12)
ax.set_ylabel('Log Play Count', fontsize=12)
ax.set_title('Log Play Count Distribution by Gender', fontsize=12)
# Customize tick labels
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
# Display the plot
plt.savefig("gender_by_popularity.png")
plt.show()

gender_counts = track_features_gender_df['gender'].value_counts()

# Calculate the percentages
gender_percentages = (gender_counts / gender_counts.sum()) * 100 # 92.7% women
# Grouping the data by genre and gender
grouped_data = track_features_gender_df.groupby(['genre', 'gender']).size().unstack(fill_value=0)

# Find the minimum count between male and female
gender_counts = track_features_gender_df['gender'].value_counts()
min_count = gender_counts.min()

# Sample the larger group to match the size of the smaller group
df_male = track_features_gender_df[track_features_gender_df['gender'] == 'male'].sample(min_count, random_state=42)
df_female = track_features_gender_df[track_features_gender_df['gender'] == 'female'].sample(min_count, random_state=42)

# Combine the balanced groups
df_balanced = pd.concat([df_male, df_female])

# Verify the new counts
balanced_gender_counts = df_balanced['gender'].value_counts()

# Grouping the data by genre and gender for the balanced dataset
grouped_data_balanced = df_balanced.groupby(['genre', 'gender']).size().unstack(fill_value=0)

# Set the plot style
plt.style.use("dark_background")

# Create the figure and axes for two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 8))

# Plot the original data
grouped_data.plot(kind='bar', stacked=False, color=['#4CAF50', '#FF6F61'], ax=ax1)
ax1.set_xlabel('Genre', fontsize=12)
ax1.set_ylabel('Count', fontsize=12)
ax1.set_title('Male/Female Split in Different Genres', fontsize=12)
ax1.tick_params(axis='x', rotation=45, labelsize=12)
ax1.tick_params(axis='y', labelsize=12)

# Plot the balanced data
grouped_data_balanced.plot(kind='bar', stacked=False, color=['#4CAF50', '#FF6F61'], ax=ax2)
ax2.set_xlabel('Genre', fontsize=12)
ax2.set_ylabel('Count', fontsize=12)
ax2.set_title('Male/Female Split in Different Genres (Balanced Dataset)', fontsize=12)
ax2.tick_params(axis='x', rotation=45, labelsize=12)
ax2.tick_params(axis='y', labelsize=12)

# Adjust layout and save the plot
plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to make room for the legend
plt.savefig("gender_and_genre_combined.png")
plt.show()



track_features_country_df = pd.merge(track_features_df, lastfm_diverse_pivot_df[['tid', 'nationalities']], 
                                  how='left', left_on='track_id', right_on='tid').drop('tid', axis=1)


non_na_both = track_features_country_df.dropna(subset=['nationalities', 'total_play_count']).shape[0]

#track_features_country_grouped_df = track_features_country_df[["nationalities","log_total_play_count"]].groupby('nationalities', as_index=False).sum()
#track_features_country_grouped_df = track_features_country_grouped_df[(track_features_country_grouped_df['nationalities'] != 'world')  &
#    (track_features_country_grouped_df['nationalities'] != 'scandinavian')]#

#track_features_country_grouped_df['iso_alpha_3'] = coco.convert(names=track_features_country_grouped_df['nationalities'], to='ISO3')
#track_features_country_grouped_df = track_features_country_grouped_df[['log_total_play_count', 'iso_alpha_3',"nationalities"]]
#unique_nationalities = track_features_country_df['nationalities'].unique()

#track_features_country_df.sort_values(by="log_total_play_count", ascending=False)
#track_features_country_grouped_df.sort_values(by="log_total_play_count", ascending=False)
#track_features_country_df[track_features_country_df["nationalities"] == "luxembourg"]

with pd.option_context('display.max_rows', 5, 'display.max_columns', None): 
    print(track_features_country_df[track_features_country_df["nationalities"] == "eswatini"])


#plt.style.use('dark_background')
## Create basic choropleth map
#fig = px.choropleth(
#    track_features_country_grouped_df,
#    locations='iso_alpha_3',
#    color='log_total_play_count',
#    color_continuous_scale='viridis',  
#    template="plotly_dark",# Color scale
#    title='Total Play Count by Country',
#    labels={'total_play_count': 'Total Play Count'},
#    hover_name='nationalities'  # Show country names on hover
#)
#plt.savefig("world_map.png")
#fig.show()#

#track_features_country_df.dropna(inplace= True)
#track_features_country_grouped_df = track_features_country_df[["nationalities","artist_familiarity"]].groupby('nationalities', as_index=False).mean()
#track_features_country_grouped_df = track_features_country_grouped_df[(track_features_country_grouped_df['nationalities'] != 'world')  &
#    (track_features_country_grouped_df['nationalities'] != 'scandinavian')]

#track_features_country_grouped_df['iso_alpha_3'] = coco.convert(names=track_features_country_grouped_df['nationalities'], to='ISO3')
#track_features_country_grouped_df = track_features_country_grouped_df[['artist_hotttnesss', 'iso_alpha_3',"nationalities"]]
#unique_nationalities = track_features_country_df['nationalities'].unique()

#track_features_country_df.sort_values(by="artist_familiarity", ascending=False)
#track_features_country_grouped_df.sort_values(by="artist_familiarity", ascending=False)
#track_features_country_df[track_features_country_df["nationalities"] == "luxembourg"]#


#plt.style.use('dark_background')
## Create basic choropleth map
#fig = px.choropleth(
#    track_features_country_grouped_df,
#    locations='iso_alpha_3',
 #   color='artist_familiarity',
#    color_continuous_scale='viridis',  
#    template="plotly_dark",# Color scale
#    title='Total Play Count by Country',
#    labels={'total_play_count': 'Total Play Count'},
#    hover_name='nationalities'  # Show country names on hover
#)
#fig.write_image("log_total_play_count_map.png")
#fig.show()



track_features_all_df = pd.merge(track_features_df, lastfm_diverse_pivot_df[['tid', 'nationalities', 'gender', 
                                                                             'language', 'religion',  'continent']],
                                  how='left', left_on='track_id', right_on='tid').drop('tid', axis=1)

track_features_all_df.to_csv(r"C:\Users\resha\data\track_features_all_df.csv")
