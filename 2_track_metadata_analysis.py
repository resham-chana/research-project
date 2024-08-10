# importing relevant libraries 
import pandas as pd
import matplotlib.pyplot as plt

# read in track metadata csv
track_metadata_df = pd.read_csv(r"C:\Users\resha\data\track_metadata_df.csv")

# remove unwanted columns 
track_metadata_cleaned_df = track_metadata_df.drop(columns=['Unnamed: 0','artist_id','artist_mbid',])

# check column titles
track_metadata_cleaned_df.columns 

# remove year = 0 as this does not make sense
track_metadata_cleaned_df = track_metadata_cleaned_df[track_metadata_cleaned_df['year'] != 0]

# get the value counts for year 
year_counts = track_metadata_cleaned_df['year'].value_counts().sort_index()

# visualise year 
plt.style.use("dark_background")
fig, ax = plt.subplots(figsize=(12, 8))
year_counts.plot(kind="bar", color="#4CAF50", ax=ax)
ax.set_xlabel('Year', fontsize = 12)
ax.set_ylabel('Count', fontsize = 12)
ax.set_title('Track Count by Year', fontsize = 12)
plt.xticks(fontsize=8)
years = year_counts.index
xticks = [i for i, year in enumerate(years) if year % 10 == 0] # label every 10 years
xtick_labels = [str(year) for year in years if year % 10 == 0]
ax.set_xticks(xticks) 
ax.set_xticklabels(xtick_labels, rotation=45, fontsize = 12)  
plt.savefig("year_distribution.png")
plt.show()
