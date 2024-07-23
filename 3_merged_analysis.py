import pandas as pd
# Seaborn visualization library
import seaborn as sns
# Create the default pairplot
import matplotlib.pyplot as plt
import numpy as np
lastfm_diverse_tags_df = pd.read_csv(r"C:\Users\corc4\data\lastfm_diverse_tags_df.csv")
lastfm_diverse_pivot_df = pd.read_csv(r"C:\Users\corc4\data\lastfm_diverse_pivot_df.csv")
track_metadata_cleaned_df = pd.read_csv(r"C:\Users\corc4\data\track_metadata_cleaned_df.csv")  
train_triplets_df = pd.read_csv(r"C:\Users\corc4\data\train_triplets_df.csv")
play_count_grouped_df = pd.read_csv(r"C:\Users\corc4\data\play_count_grouped_df.csv")
genres_df = pd.read_csv(r"C:\Users\corc4\data\genres_df.csv")

#images_df = pd.read_csv(r'C:\Users\corc4\Downloads\MSD-I_dataset.tsv', sep='\t')
#images_df.columns
#images_df = images_df.iloc[:,[0,1,2]]


track_features_df = pd.merge(track_metadata_cleaned_df, play_count_grouped_df.iloc[:,[1,2]], left_on='song_id', right_on='song').drop('song', axis=1)
track_features_df = pd.merge(track_features_df, genres_df.iloc[:,[1,2]], how='inner', on='track_id')
track_features_df.dropna(inplace= True)
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

track_features_gender_df.dropna(inplace= True)
#'track_id', 'title', 'song_id', 
#'release', 'artist_name', 'duration',
#'artist_familiarity', 'artist_hotttnesss', 
#'year', 'total_play_count','genre', "country","language","gender", ""


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




track_features_country_df = pd.merge(track_features_df, lastfm_diverse_pivot_df[['tid', 'nationalities']], 
                                  how='left', left_on='track_id', right_on='tid').drop('tid', axis=1)


track_features_country_df.dropna(inplace= True)
track_features_country_grouped_df = track_features_country_df.groupby('nationalities', as_index=False).sum()
track_features_country_grouped_df = track_features_country_grouped_df[track_features_country_grouped_df['nationalities'] != 'world']

track_features_country_grouped_df['iso_alpha_3'] = coco.convert(names=track_features_country_grouped_df['nationalities'], to='ISO3')
track_features_country_grouped_df = track_features_country_grouped_df[['total_play_count', 'iso_alpha_3',"nationalities"]]

import plotly.express as px
import country_converter as coco


# Create basic choropleth map
fig = px.choropleth(
    track_features_country_grouped_df,
    locations='iso_alpha_3',
    color='total_play_count',
    color_continuous_scale='OrRd',  # Color scale
    title='Total Play Count by Country',
    labels={'total_play_count': 'Total Play Count'},
    hover_name='nationalities'  # Show country names on hover
)
fig.show()