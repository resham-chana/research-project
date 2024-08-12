# import relevant libraries 
import pandas as pd
import re
import numpy as np
import plotly.express as px
import country_converter as coco

# read in data used
train_triplets_df = pd.read_csv(r"C:\Users\resha\data\train_triplets_df.csv")
MSD_df = pd.read_csv(r"C:\Users\resha\data\MSD_subset.csv")
play_count_grouped_df = pd.read_csv(r"C:\Users\resha\data\play_count_grouped_df.csv")
genres_df = pd.read_csv(r"C:\Users\resha\data\genres_df.csv")
lastfm_diverse_pivot_df = pd.read_csv(r"C:\Users\resha\data\lastfm_diverse_pivot_df.csv")
geography_df = pd.read_csv(r"C:\Users\resha\data\geography_df.csv")

# define a regex to match byte types
pattern = re.compile(r"b'(.*?)'")

def extract_if_bytes(x):
    '''
    Function to return the conent inside a byte string
    '''
    if isinstance(x, str):
        # if pattern matches string return the content 
        match = pattern.search(x)
        if match:
            return match.group(1) 
    return x  

# apply function to song_id and track_id
for col in ['song_id', 'track_id']:
    if col in MSD_df.columns:
        MSD_df[col] = MSD_df[col].apply(extract_if_bytes)

MSD_df["artist_location"].nunique()

unique_artist_locations = MSD_df["artist_location"].unique()

# print all unique artist locations
print("Unique artist locations:")
for location in unique_artist_locations:
    print(location)

# states from [https://www.drupal.org/node/332575]
states = [('Alabama', 'AL'),('Kentucky', 'KY'),('Ohio', 'OH'),('Alaska', 'AK'), ('Louisiana', 'LA'),('Oklahoma', 'OK'), ('Arizona', 'AZ'),('Maine', 'ME')
 ,('Oregon', 'OR'),('Arkansas', 'AR'),('Maryland', 'MD'),('Pennsylvania', 'PA'),('American Samoa', 'AS'),
 ('Massachusetts', 'MA'),('Puerto Rico', 'PR'),('California', 'CA'),('Michigan', 'MI'),('Rhode Island', 'RI'),('Colorado', 'CO'), ('Minnesota', 'MN'),('South Carolina', 'SC'),
 ('Connecticut', 'CT'),('Mississippi', 'MS'),('South Dakota', 'SD'),('Delaware', 'DE'),('Missouri', 'MO'),('Tennessee', 'TN'),('District of Columbia', 'DC'),
 ('Montana', 'MT'),('Texas', 'TX'),('Florida', 'FL'),('Nebraska', 'NE'),('Trust Territories', 'TT'),('Georgia', 'GA'),('Nevada', 'NV'),('Utah', 'UT'),('Guam', 'GU'),('New Hampshire', 'NH'),
 ('Vermont', 'VT'),('Hawaii', 'HI'),('New Jersey', 'NJ'),('Virginia', 'VA'),('Idaho', 'ID'),('New Mexico', 'NM'),('Virgin Islands', 'VI'),('Illinois', 'IL'),('New York', 'NY'),
 ('Washington', 'WA'),('Indiana', 'IN'),('North Carolina', 'NC'),('West Virginia', 'WV'),('Iowa', 'IA'),('North Dakota', 'ND'),('Wisconsin', 'WI'),('Kansas', 'KS'),('Northern Mariana Islands', 'MP'),
 ('Wyoming', 'WY')
]

# take countries from cia factbook dataset
countries = geography_df["country"]


def determine_geo_msd(tag, states, countries):
    '''
    function to 
    determine a country from a tag
    '''
    def space_front_word(word, text):
        '''
        helper function to check if a word is not a substring of another word
        '''
        pattern = r'\b' + re.escape(word) + r'\b'
        return bool(re.search(pattern, text, re.IGNORECASE))
    if 'england' in tag.lower() or 'wales' in tag.lower() or 'scotland' in tag.lower():
        return "United Kingdom"
    for state, abbreviation in states:
        if space_front_word(state, tag) or space_front_word(abbreviation, tag):
            return "United States" # return united states if the tag is a state
    for country in countries:
        if space_front_word(country, tag):
            return country

    return None

# apply function to the MSD_df
unique_locations = MSD_df["artist_location"].unique()
country_mapping = {tag: determine_geo_msd(tag, states, countries) for tag in unique_locations}
MSD_df['country'] = MSD_df['artist_location'].map(country_mapping)
print(MSD_df[['country', 'artist_location']]).drop_duplicates()

# create a new df of mismatches
none_in_country = MSD_df[MSD_df['country'].isnull()]
missing_countries = none_in_country.drop_duplicates(subset=['artist_location'])[['artist_location', 'country']]
# save to csv
missing_countries.to_csv(r"C:\Users\resha\data\missing_countries_df.csv")
missing_countries_match = pd.read_csv(r"C:\Users\resha\data\missing_countries_match_df.csv")

# deal with encoding issues by trying both utf-8 and latin1
try:
    missing_countries_match = pd.read_csv(r"C:\Users\resha\data\missing_countries_match_df.csv", encoding='utf-8')
except UnicodeDecodeError:
    missing_countries_match = pd.read_csv(r"C:\Users\resha\data\missing_countries_match_df.csv", encoding='latin1')

# create a dictionary for country and artist location and fill NAs in MSD_df
location_to_country_dict = missing_countries_match.set_index('artist_location')['country'].to_dict()
MSD_df['country'] = MSD_df['country'].fillna(MSD_df['artist_location'].map(location_to_country_dict))


# merge dataset with genre and playcount
MSD_merged_df = pd.merge(MSD_df, play_count_grouped_df.iloc[:,[1,2]], left_on='song_id', right_on='song', how = "left").drop('song', axis=1)
MSD_merged_df['log_total_play_count'] = np.log10(MSD_merged_df['total_play_count'])
non_na_both = MSD_merged_df.dropna(subset=['country', 'total_play_count']).shape[0]

MSD_df.columns
checkrows = MSD_df.dropna(subset="country")
MSD_grouped_df = MSD_merged_df[["country","log_total_play_count"]].groupby('country', as_index=False).sum()
MSD_grouped_df = MSD_grouped_df.dropna(subset="country")

MSD_grouped_df['iso_alpha_3'] = coco.convert(names=MSD_grouped_df['country'], to='ISO3')
MSD_grouped_df = MSD_grouped_df[['log_total_play_count', 'iso_alpha_3',"country"]]

# visualise log play count and country with a world map
fig = px.choropleth(
    MSD_grouped_df,
    locations='iso_alpha_3',
    color='log_total_play_count',
    color_continuous_scale='viridis',  
    template="plotly_dark",
    title='Total Log Play Count by Country',
    labels={'log_total_play_count': 'Total Play Count'},
    hover_name='country'  
)
fig.write_image("log_total_play_count_map.png")
fig.show()

# merge MSD subset with genre for potential analysis
MSD_merged_df= pd.merge(MSD_merged_df, genres_df.iloc[:,[1,2]], how='inner', on='track_id')
MSD_merged_df.to_csv(r"C:\Users\resha\data\MSD_merged_df.csv")
