# import relevant libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

lastfm_diverse_tags_df = pd.read_csv(r"C:\Users\resha\data\lastfm_diverse_tags_df.csv")
lastfm_diverse_pivot_df = pd.read_csv(r"C:\Users\resha\data\lastfm_diverse_pivot_df.csv")
track_metadata_df = pd.read_csv(r"C:\Users\resha\data\track_metadata_df.csv")
train_triplets_df = pd.read_csv(r"C:\Users\resha\data\train_triplets_df.csv")
play_count_grouped_df = pd.read_csv(r"C:\Users\resha\data\play_count_grouped_df.csv")
genres_df = pd.read_csv(r"C:\Users\resha\data\genres_df.csv")

# create new dataframe with metadata, total play count and genre
track_features_df = pd.merge(track_metadata_df, play_count_grouped_df.iloc[:,[1,2]], left_on='song_id', right_on='song', how ="left").drop('song', axis=1)
track_features_df = pd.merge(track_features_df, genres_df.iloc[:,[1,2]], how='left', on='track_id')
# add in log total play count because the data is quite skewed
track_features_df['log_total_play_count'] = np.log10(track_features_df['total_play_count'])
selected_vars = ['duration','year', 'artist_familiarity', 'artist_hotttnesss'
                 ,'log_total_play_count']

# create a pairwise plot
plt.style.use("dark_background")
sns.pairplot(track_features_df, vars=selected_vars, hue = "genre",  palette="husl"
             ,kind = "kde",plot_kws={'alpha':0.5})
plt.savefig("pairwiseplot.png")
plt.show()

# join the gender column from the last fm tags
track_features_gender_df = pd.merge(track_features_df, lastfm_diverse_pivot_df[['tid', 'gender']], 
                                  how='left', left_on='track_id', right_on='tid').drop('tid', axis=1)

# visualise the gender as a boxplot
plt.style.use("dark_background")
fig, ax = plt.subplots(figsize=(12, 8))
sns.boxplot(x='gender', y='log_total_play_count', data=track_features_gender_df,
             palette='viridis', ax=ax)
ax.set_xlabel('Gender', fontsize=12)
ax.set_ylabel('Log Play Count', fontsize=12)
ax.set_title('Log Play Count Distribution by Gender', fontsize=12)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.savefig("gender_by_popularity.png")
plt.show()

# counts for male and female
gender_counts = track_features_gender_df['gender'].value_counts()

# calculate the percentages
gender_percentages = (gender_counts / gender_counts.sum()) * 100 # 92.7% women
# grouping the data by genre and gender
grouped_data = track_features_gender_df.groupby(['genre', 'gender']).size().unstack(fill_value=0)
# find the minimum count between male and female
min_count = gender_counts.min()

# resample so that the number of men and women are the same
df_male = track_features_gender_df[track_features_gender_df['gender'] == 'male'].sample(min_count, random_state=42)
df_female = track_features_gender_df[track_features_gender_df['gender'] == 'female'].sample(min_count, random_state=42)

# concatenate the two dataframes
df_balanced = pd.concat([df_male, df_female])
balanced_gender_counts = df_balanced['gender'].value_counts()

# group the data by genre and gender 
grouped_data_balanced = df_balanced.groupby(['genre', 'gender']).size().unstack(fill_value=0)

# visualise gender and genre
plt.style.use("dark_background")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 8))

# original data
grouped_data.plot(kind='bar', stacked=False, color=['#4CAF50', '#FF6F61'], ax=ax1)
ax1.set_xlabel('Genre', fontsize=12)
ax1.set_ylabel('Count', fontsize=12)
ax1.set_title('Male/Female Split in Different Genres', fontsize=12)
ax1.tick_params(axis='x', rotation=45, labelsize=12)
ax1.tick_params(axis='y', labelsize=12)

# balanced data
grouped_data_balanced.plot(kind='bar', stacked=False, color=['#4CAF50', '#FF6F61'], ax=ax2)
ax2.set_xlabel('Genre', fontsize=12)
ax2.set_ylabel('Count', fontsize=12)
ax2.set_title('Male/Female Split in Different Genres (Balanced Dataset)', fontsize=12)
ax2.tick_params(axis='x', rotation=45, labelsize=12)
ax2.tick_params(axis='y', labelsize=12)
plt.tight_layout(rect=[0, 0, 1, 0.95])  
plt.savefig("gender_and_genre_combined.png")
plt.show()

track_features_all_df = pd.merge(track_features_df, lastfm_diverse_pivot_df[['tid', 'nationalities', 'gender', 
                                                                             'language', 'religion',  'continent']],
                                  how='left', left_on='track_id', right_on='tid').drop('tid', axis=1)

track_features_all_df.to_csv(r"C:\Users\resha\data\track_features_all_df.csv")
