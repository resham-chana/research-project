import pandas as pd
import re
import numpy as np
import plotly.express as px
import country_converter as coco

train_triplets_df = pd.read_csv(r"C:\Users\resha\data\train_triplets_df.csv")
MSD_df = pd.read_csv(r"C:\Users\resha\data\MSD_subset.csv")
play_count_grouped_df = pd.read_csv(r"C:\Users\resha\data\play_count_grouped_df.csv")
genres_df = pd.read_csv(r"C:\Users\resha\data\genres_df.csv")
lastfm_diverse_pivot_df = pd.read_csv(r"C:\Users\resha\data\lastfm_diverse_pivot_df.csv")
geography_df = pd.read_csv(r"C:\Users\resha\data\geography_df.csv")
# Define a function to decode bytes

pattern = re.compile(r"b'(.*?)'")

# Function to apply regex and extract matched content if present
def extract_if_bytes(x):
    if isinstance(x, str):
        # Apply regex to extract content if pattern matches
        match = pattern.search(x)
        if match:
            return match.group(1)  # Return the first matched group (content inside b'')
    return x  # Return unchanged if no match or not a string

# Apply regex to each column where dtype is object
for col in ['song_id', 'track_id']:
    if col in MSD_df.columns:
        MSD_df[col] = MSD_df[col].apply(extract_if_bytes)

MSD_df["artist_location"].nunique()

unique_artist_locations = MSD_df["artist_location"].unique()

# Print all unique artist locations
print("Unique artist locations:")
for location in unique_artist_locations:
    print(location)

states = [('Alabama', 'AL'),('Kentucky', 'KY'),('Ohio', 'OH'),('Alaska', 'AK'), ('Louisiana', 'LA'),('Oklahoma', 'OK'), ('Arizona', 'AZ'),('Maine', 'ME')
 ,('Oregon', 'OR'),('Arkansas', 'AR'),('Maryland', 'MD'),('Pennsylvania', 'PA'),('American Samoa', 'AS'),
 ('Massachusetts', 'MA'),('Puerto Rico', 'PR'),('California', 'CA'),('Michigan', 'MI'),('Rhode Island', 'RI'),('Colorado', 'CO'), ('Minnesota', 'MN'),('South Carolina', 'SC'),
 ('Connecticut', 'CT'),('Mississippi', 'MS'),('South Dakota', 'SD'),('Delaware', 'DE'),('Missouri', 'MO'),('Tennessee', 'TN'),('District of Columbia', 'DC'),
 ('Montana', 'MT'),('Texas', 'TX'),('Florida', 'FL'),('Nebraska', 'NE'),('Trust Territories', 'TT'),('Georgia', 'GA'),('Nevada', 'NV'),('Utah', 'UT'),('Guam', 'GU'),('New Hampshire', 'NH'),
 ('Vermont', 'VT'),('Hawaii', 'HI'),('New Jersey', 'NJ'),('Virginia', 'VA'),('Idaho', 'ID'),('New Mexico', 'NM'),('Virgin Islands', 'VI'),('Illinois', 'IL'),('New York', 'NY'),
 ('Washington', 'WA'),('Indiana', 'IN'),('North Carolina', 'NC'),('West Virginia', 'WV'),('Iowa', 'IA'),('North Dakota', 'ND'),('Wisconsin', 'WI'),('Kansas', 'KS'),('Northern Mariana Islands', 'MP'),
 ('Wyoming', 'WY')
]

countries = geography_df["country"]

def determine_geo_msd(tag, states, countries):
    def space_front_word(word, text):
        pattern = r'\b' + re.escape(word) + r'\b'
        return bool(re.search(pattern, text, re.IGNORECASE))

    # Check if 'England' is in the tag
    if 'england' in tag.lower() or 'wales' in tag.lower() or 'scotland' in tag.lower():
        return "United Kingdom"

    # Check against states first
    for state, abbreviation in states:
        if space_front_word(state, tag) or space_front_word(abbreviation, tag):
            return "United States"

    # Check against countries
    for country in countries:
        if space_front_word(country, tag):
            return country

    return None

# Applying the function to the DataFrame
unique_locations = MSD_df["artist_location"].unique()
country_mapping = {tag: determine_geo_msd(tag, states, countries) for tag in unique_locations}

# Map the country back to the original DataFrame
MSD_df['country'] = MSD_df['artist_location'].map(country_mapping)

# Print the specified columns
print(MSD_df[['country', 'artist_location']])

unique = MSD_df.drop_duplicates(subset=['artist_location'])
with pd.option_context('display.max_rows', None,):
    print(unique[['artist_location', 'country']])

play_count_songs = set(play_count_grouped_df['song'])
msd_songs = set(MSD_df['song_id'])

# Find common songs
common_songs = play_count_songs.intersection(msd_songs)

MSD_merged_df = pd.merge(MSD_df, play_count_grouped_df.iloc[:,[1,2]], left_on='song_id', right_on='song').drop('song', axis=1)
MSD_merged_df = pd.merge(MSD_merged_df, genres_df.iloc[:,[1,2]], how='inner', on='track_id')
MSD_merged_df['log_total_play_count'] = np.log10(MSD_merged_df['total_play_count'])

MSD_df.columns
checkrows = MSD_merged_df.dropna(subset="country")
MSD_grouped_df = MSD_merged_df[["country","log_total_play_count"]].groupby('country', as_index=False).mean()
MSD_grouped_df = MSD_grouped_df.dropna(subset="country")

MSD_grouped_df['iso_alpha_3'] = coco.convert(names=MSD_grouped_df['nationalities'], to='ISO3')
track_features_country_grouped_df = MSD_grouped_df[['log_total_play_count', 'iso_alpha_3',"country"]]
#unique_nationalities = MSD_grouped_df['country'].unique()

track_features_country_df.sort_values(by="artist_familiarity", ascending=False)
track_features_country_grouped_df.sort_values(by="artist_familiarity", ascending=False)
track_features_country_df[track_features_country_df["nationalities"] == "luxembourg"]


plt.style.use('dark_background')
# Create basic choropleth map
fig = px.choropleth(
    track_features_country_grouped_df,
    locations='iso_alpha_3',
    color='artist_familiarity',
    color_continuous_scale='viridis',  
    template="plotly_dark",# Color scale
    title='Total Play Count by Country',
    labels={'total_play_count': 'Total Play Count'},
    hover_name='nationalities'  # Show country names on hover
)
fig.write_image("log_total_play_count_map.png")
fig.show()
