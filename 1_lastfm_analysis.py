# install relevant packages
import numpy as np
import matplotlib
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from thefuzz import fuzz
from thefuzz import process
from wordcloud import WordCloud
import seaborn as sns
import os
import re
from sklearn.metrics.pairwise import cosine_similarity

# read processed lastfm tag datasets
lastfmpath = r"C:\Users\resha\data\lastfm_tags_df.csv"
pivot_path = r"C:\Users\resha\data\lastfm_pivot_df.csv"
country_path = r"C:\Users\resha\data\geography_df.csv"
#lastfmpath = r"C:\Users\corc4\data\lastfm_tags_df.csv"
#pivot_path = r"C:\Users\corc4\data\lastfm_pivot_df.csv"

lastfm_tags_df = pd.read_csv(lastfmpath)
lastfm_pivot_df = pd.read_csv(pivot_path)

# visualise most common tags
tag_counts = lastfm_tags_df["tag"].value_counts()

# convert to dataframe 
tag_counts_df = tag_counts.reset_index()
tag_counts_df.columns = ['tag', 'count']
tag_counts_df = tag_counts_df.head(20).sort_values("count", ascending=True)

plt.style.use("dark_background")
fig, ax = plt.subplots(figsize=(12,8))
bar_container = ax.barh("tag", "count", data=tag_counts_df, color="#4CAF50")
plt.title('Top 20 Tags by Count')
plt.xlabel('Count')
plt.ylabel('Tag')
plt.yticks(fontsize=11.9)
ax.get_xaxis().set_major_formatter(
     mtick.FuncFormatter(lambda x, p: format(int(x), ','))) # add commas to numbers
labels = ax.bar_label(bar_container, fmt='{:,.0f}', color='white')
for label in labels:
    label.set_fontsize(13)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.savefig(r"C:\Users\resha\plots\popular_tags.png")
#plt.savefig(r"C:\Users\corc4\plots\popular_tags.png")

plt.show()


# cleaning tags
lastfm_tags_df['tag'] = lastfm_tags_df['tag'].astype(str) # string type
# remove unwanted punctuation and to lowercase
lastfm_tags_df['cleaned_tag'] = lastfm_tags_df['tag'].str.replace(r'[^a-zA-Z0-9\s]', '', regex=True).str.strip().str.lower()
lastfm_tags_df = lastfm_tags_df[lastfm_tags_df['cleaned_tag'].str.len() > 1] # remove short tags
unique_tags = lastfm_tags_df['cleaned_tag'].unique()

############################################# fuzzy grouping gender #####################################################

# Define the words to match
female_terms = ['woman', 'female', 'female singer', 'female vocalist', 'female vocal', 'female voice']
male_terms = ['man', 'male','male singer', 'male vocalist']

def is_standalone_word(word, text):
    # standalone word if tere are no other letters surrounding it:
    pattern = r'\b' + re.escape(word) + r'\b'
    matches = re.findall(pattern, text)
    return bool(matches)

def determine_gender(tag):
    # scores for female and male terms
    female_score = 0
    male_score = 0
    
    # calculate fuzzy score for female
    for female_term in female_terms:
        score = fuzz.ratio(tag, female_term)
        if is_standalone_word(female_term, tag):
            female_score = max(female_score, score)
    
    # calculate fuzzy score for male
    for male_term in male_terms:
        score = fuzz.ratio(tag, male_term)
        if is_standalone_word(male_term, tag):
            male_score = max(male_score, score)

    # decide gender based on highest score
    if male_score > female_score:
        return 'male'
    elif female_score > male_score:
        return 'female'
    else:
        return 'NaN'

# group by cleaned_tag and apply the gender function
unique_tags = lastfm_tags_df['cleaned_tag'].unique()
gender_mapping = {tag: determine_gender(tag) for tag in unique_tags}

# map the gender back to the original dataframe
lastfm_tags_df['gender'] = lastfm_tags_df['cleaned_tag'].map(gender_mapping)

# what tags are matched as male and female - do they make sense?
female_df = lastfm_tags_df[lastfm_tags_df['gender'] == 'female']      
female_df.value_counts(subset=['cleaned_tag']).head(50)
male_df = lastfm_tags_df[lastfm_tags_df['gender'] == 'male']
male_df.value_counts(subset=['cleaned_tag']).head(50)
unknown_gender_df = lastfm_tags_df[lastfm_tags_df['gender'] == 'NaN'] 
unknown_gender_df.value_counts(subset=['cleaned_tag'])

# what percentage has been labelled?
(female_df["tid"].nunique() + male_df["tid"].nunique())/ (lastfm_tags_df["tid"].nunique())
# visualise the wordclouds
gender_counts = lastfm_tags_df['gender'].value_counts() # 19% of unique tracks (may be repeats in labelling)

# word cloud to display whihc tags fit with female
for gender in gender_counts.index:
    wordcloud = WordCloud(width=800, height=400, background_color='black', colormap='Greens').generate(' '.join(lastfm_tags_df[lastfm_tags_df['gender'] == gender]['cleaned_tag']))
    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title(f'Word Cloud for {gender}')
    plt.axis('off')
    # Save the word cloud to a file
    wordcloud.to_file(os.path.join(r"C:\Users\resha\plots", f"wordcloud_{gender}.png"))
    #wordcloud.to_file(os.path.join(r"C:\Users\corc4\plots", f"wordcloud_{gender}.png"))

    plt.show()

################################################ fuzzy grouping countries/languages/ethnicities/religion #######################################################

# checking if the tags countain different languages:
#geography_path = r"C:\Users\resha\data\geography_df.csv"
#geography_df = pd.read_csv(geography_path)
#geography_path = r"C:\Users\corc4\data\geography_df.csv"
geography_df = pd.read_csv(country_path)

# function to remove words in bracks
def clean(s):
    # things in brackets
    s = re.sub(r'\(.*?\)', '', s)
    # Remove everything after 'note'
    s = re.sub(r'note.*', '', s, flags=re.IGNORECASE)
    return s

# add new data
new_data = pd.DataFrame({
    "country": ["united kingdom", "united kingdom", "united states", "scandinavian"],
    "nationality": ["brit", "uk", "usa", "scandi"]
})


nationality_df = geography_df[['country', 'nationality']].astype(str) # to string
nationality_df['country'] = nationality_df['country'].apply(clean).str.lower() # to lower
nationality_df['nationality'] = nationality_df['nationality'].apply(clean).str.lower()  # to lower
nationality_df['country'] = nationality_df['country'].replace({'north korean': 'korean', 'south korean': 'korean'}) # alter korea to properly label it
nationality_df['nationality'] = nationality_df['nationality'].apply(lambda x: re.split(r'\s* or \s*|\s*;\s*|\s*,\s*', x))# split nationalities 
nationality_df = nationality_df.explode('nationality', ignore_index=True)
nationality_df['nationality'] = nationality_df['nationality'].str.strip() # remove unwanted spaces
nationality_df = nationality_df[nationality_df["nationality"] != '']
nationality_df["nationality"] = nationality_df["nationality"].replace(['nan', 'none',"NaN"], pd.NA).fillna(nationality_df["country"]) # remove NaNs
nationality_df = nationality_df[~((nationality_df['nationality'] == 'dutch') & (nationality_df['country'] != 'netherlands'))] # alter dutch labelling
nationality_df = pd.concat([nationality_df, new_data], ignore_index=True) # add new data

# create list of nationalities
nationalities = list(nationality_df.to_records(index = False))

############################ fuzzy grouping language ##############################

languages = ["mandarin", "chinese", "spanish", "english", "hindi", "bengali", "portuguese", "russian", "japanese", "vietnamese", 
             "turkish", "marathi", "telugu", "punjabi", "korean", "tamil", "german", "french", "urdu", "arabic", 
             "javanese", "italian", "iranian persian", "gujarati", "hausa", "bhojpuri", "southern min"]

def determine_language(tag):
    highest_score = 0
    best_match = "NaN"
    
    # check if any language is a standalone word in the tag
    for language in languages:
        if is_standalone_word(language, tag):
            return language
    # calculate fuzzy score
    for language in languages:
        score = fuzz.ratio(tag, language)
        if score > highest_score:
            highest_score = score
            best_match = language
    
    return best_match if highest_score > 90 else "NaN"

# apply language function
unique_tags = lastfm_tags_df['cleaned_tag'].unique()
language_mapping = {tag: determine_language(tag) for tag in unique_tags}

# map this onto dataframe
lastfm_tags_df['language'] = lastfm_tags_df['cleaned_tag'].map(language_mapping)

# show df
lastfm_tags_df.head(50)
lastfm_tags_df[lastfm_tags_df['language'] != 'NaN']

######################################### fuzzy grouping religion/continents ########################################

religions = [("christianity", "christian"),("islam", "muslim"),("hinduism", "hindu"),
             ("buddhism", "buddhist"),("sikhism", "sikh"),("judaism", "jewish"),("bahai","bahais"), ("jainism", "jain"),("shinto", "shintoist"),("cao dai", "caodaism"),
             ("zoroastrianism", "zoroastrian"),("tenrikyo","tenrikyos"),("animism", "animist"),
               ("paganism", "pagan"), ("unitarian universalism", "unitarian universalist"),("rastafari", "rastafarian")]

continents = [("asia","asian"),("north america","american"),("south america","latin"),("europe","european"),("australia","aussie"),("africa","african")]

def determine_geo(tag, tuples):
    # List of standalone words to exclude
    # Regular expression to check if 'oman' is not surrounded by letters (i.e., it's a standalone word)
    def is_standalone_word(word, text):
        # Check for exact match
        pattern = r'\b' + re.escape(word) + r'\b'
        matches = re.findall(pattern, text)
        if matches:
            return bool(matches)
    for geo, alternative_name in tuples:
        # Regular expression to check if 'oman' is not surrounded by letters
 #       if geo in tag or alternative_name in tag:
        if is_standalone_word(geo, tag) or is_standalone_word(alternative_name,tag):
            return geo

        # Calculate scores for both geo and its alternative name
        score_geo = fuzz.ratio(tag, geo)
        score_alternative = fuzz.ratio(tag, alternative_name)
        highest_score = 0
        # Determine the highest score and the best match
        if score_geo > highest_score:
            highest_score = score_geo
            best_match = geo
        if score_alternative > highest_score:
            highest_score = score_alternative
            best_match = alternative_name
    
    return best_match if highest_score > 90 else "NaN"

# Group by cleaned_tag and apply the religion determination function
unique_tags = lastfm_tags_df['cleaned_tag'].unique()
religion_mapping = {tag: determine_geo(tag,religions) for tag in unique_tags}
continent_mapping = {tag: determine_geo(tag,continents) for tag in unique_tags}
nationality_mapping = {tag: determine_geo(tag,nationalities) for tag in unique_tags}

# Map the gender back to the original DataFrame
lastfm_tags_df['religion'] = lastfm_tags_df['cleaned_tag'].map(religion_mapping)
lastfm_tags_df['continent'] = lastfm_tags_df['cleaned_tag'].map(continent_mapping)
lastfm_tags_df['nationalities'] = lastfm_tags_df['cleaned_tag'].map(nationality_mapping)

# Display the DataFrame
lastfm_tags_df.head(50)
lastfm_tags_df[lastfm_tags_df['religion'] != 'NaN']
lastfm_tags_df[lastfm_tags_df['continent'] != 'NaN']
lastfm_tags_df[lastfm_tags_df['nationalities'] != 'NaN']

lastfm_diverse_tags_df = lastfm_tags_df

#lastfm_diverse_tags_df = pd.read_csv(r"C:\Users\resha\data\lastfm_diverse_tags_df.csv")

# clean up indexing
#lastfm_diverse_tags_df = lastfm_diverse_tags_df.drop(columns=['Unnamed: 0.1', 'Unnamed: 0'])
lastfm_diverse_tags_df = lastfm_diverse_tags_df.drop(columns=['Unnamed: 0'])

# fill gender/language/religion/continent/nationality for the same tracks
def fill_diverse(df, column):
    non_nan_values = df.loc[df[column].notna(), ['tid', column]]
    duplicated_values = non_nan_values.set_index('tid')[column].to_dict()
    df.loc[:, column] = df['tid'].map(duplicated_values).combine_first(df[column])
    return df

# List of columns to duplicate values for
columns_to_duplicate = ['language', 'religion', 'continent', 'nationalities']

# Apply duplication function to each specified column
for column in columns_to_duplicate:
    df = fill_diverse(lastfm_diverse_tags_df, column)

# average count per cleaned tag
cleaned_tag_counts = lastfm_diverse_tags_df['cleaned_tag'].value_counts()

cleaned_tag_counts.mean()

# remove tags if they have been used less than 16 times
tags_to_remove = cleaned_tag_counts[cleaned_tag_counts <= 17].index
lastfm_diverse_tags_df = lastfm_diverse_tags_df[~lastfm_diverse_tags_df['cleaned_tag'].isin(tags_to_remove)]

# pivoting tags 
lastfm_diverse_tags_df.loc[:, 'tag_number'] = lastfm_diverse_tags_df.groupby('tid').cumcount() + 1
lastfm_diverse_pivot_df = lastfm_diverse_tags_df.pivot(index='tid', columns='tag_number', values='tag').reset_index()
lastfm_diverse_pivot_df.columns = ['tid'] + [f'tag{i}' for i in range(1, len(lastfm_diverse_pivot_df.columns))]

# Merge the additional columns back to the pivoted DataFrame
lastfm_diverse_pivot_df = pd.merge(lastfm_diverse_pivot_df, 
                                   lastfm_diverse_tags_df[['tid', 'gender', 'language', 'religion', 'continent', 'nationalities']].drop_duplicates(subset=['tid'])
                                   , on='tid', how='left')


#lastfm_diverse_tags_df.to_csv(r"C:\Users\corc4\data\lastfm_diverse_tags_df.csv")  
#lastfm_diverse_pivot_df.to_csv(r"C:\Users\corc4\data\lastfm_diverse_pivot_df.csv")  
lastfm_diverse_tags_df.to_csv(r"C:\Users\resha\data\lastfm_diverse_tags_df.csv")  
lastfm_diverse_pivot_df.to_csv(r"C:\Users\resha\data\lastfm_diverse_pivot_df.csv")  