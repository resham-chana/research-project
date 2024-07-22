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
import gensim.downloader as api
from sklearn.metrics.pairwise import cosine_similarity

# read processed lastfm tag datasets
lastfmpath = r"C:\Users\resha\data\lastfm_tags_df.csv"
pivot_path = r"C:\Users\resha\data\lastfm_pivot_df.csv"
country_path = r"C:\Users\resha\data\country_df.csv"
#lastfmpath = r"C:\Users\corc4\data\lastfm_tags_df.csv"
#pivot_path = r"C:\Users\corc4\data\lastfm_pivot_df.csv"
 
lastfm_tags_df = pd.read_csv(lastfmpath)
lastfm_pivot_df = pd.read_csv(pivot_path)
country_df = pd.read_csv(country_path)

# visualise most common tags
tag_counts = lastfm_tags_df["tag"].value_counts()

# convert to DataFrame 
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
     mtick.FuncFormatter(lambda x, p: format(int(x), ',')))
labels = ax.bar_label(bar_container, fmt='{:,.0f}', color='white')
for label in labels:
    label.set_fontsize(13)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.savefig(r"C:\Users\resha\plots\popular_tags.png")
#plt.savefig(r"C:\Users\corc4\plots\popular_tags.png")

plt.show()


# cleaning tags

lastfm_tags_df['tag'] = lastfm_tags_df['tag'].astype(str)
lastfm_tags_df['cleaned_tag'] = lastfm_tags_df['tag'].str.replace(r'[^a-zA-Z0-9\s]', '', regex=True).str.strip().str.lower()
lastfm_tags_df = lastfm_tags_df[lastfm_tags_df['cleaned_tag'].str.len() > 1]
unique_tags = lastfm_tags_df['cleaned_tag'].unique()

############################################# fuzzy grouping gender #####################################################

# Define the words to match
female_terms = ['woman', 'female', 'female singer', 'female vocalist', 'female vocal', 'female voice']
male_terms = ['man', 'male','male singer', 'male vocalist']


def is_standalone_word(word, text):
    # Create a regular expression pattern to check if the word is a standalone word
    pattern = r'\b' + re.escape(word) + r'\b'
    matches = re.findall(pattern, text)
    return bool(matches)

def determine_gender(tag):
    # Initialize scores for female and male terms
    female_score = 0
    male_score = 0
    
    # Calculate fuzzy match scores for female terms
    for female_term in female_terms:
        score = fuzz.ratio(tag, female_term)
        if is_standalone_word(female_term, tag):
            female_score = max(female_score, score)
    
    # Calculate fuzzy match scores for male terms
    for male_term in male_terms:
        score = fuzz.ratio(tag, male_term)
        if is_standalone_word(male_term, tag):
            male_score = max(male_score, score)

    # Decide the gender based on the highest score
    if male_score > female_score:
        return 'male'
    elif female_score > male_score:
        return 'female'
    else:
        return 'NaN'

# Group by cleaned_tag and apply the gender determination function
unique_tags = lastfm_tags_df['cleaned_tag'].unique()
gender_mapping = {tag: determine_gender(tag) for tag in unique_tags}

# Map the gender back to the original DataFrame
lastfm_tags_df['gender'] = lastfm_tags_df['cleaned_tag'].map(gender_mapping)

# Display the DataFrame
print(lastfm_tags_df)

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
gender_counts = lastfm_tags_df['gender'].value_counts()

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
geography_path = r"C:\Users\resha\data\geography_df.csv"
geography_df = pd.read_csv(geography_path)

def clean_and_split(s):
    # things in brackets
    s = re.sub(r'\(.*?\)', '', s)
    # Remove everything after 'note'
    s = re.sub(r'note.*', '', s, flags=re.IGNORECASE)
    return s

new_data = pd.DataFrame({
    "country": ["united kingdom", "united kingdom", "america", "scandinavian"],
    "nationality": ["brit", "uk", "usa", "scandi"]
})

# Concatenate the old DataFrame with the new one
nationality_df = geography_df[['country', 'nationality']].astype(str)
nationality_df['country'] = nationality_df['country'].str.lower()
nationality_df['nationality'] = nationality_df['nationality'].apply(clean_and_split).str.lower()
nationality_df['nationality'] = nationality_df['nationality'].apply(lambda x: re.split(r'\s* or \s*|\s*;\s*|\s*,\s*', x))
nationality_df = nationality_df.explode('nationality', ignore_index=True)
nationality_df['nationality'] = nationality_df['nationality'].str.strip()
nationality_df = nationality_df[nationality_df["nationality"] != '']
nationality_df["nationality"] = nationality_df["nationality"].replace(['nan', 'none'], pd.NA).fillna(nationality_df["country"])
nationality_df["nationality"] = nationality_df["nationality"].drop([14,69,112,137])
nationality_df = pd.concat([nationality_df, new_data], ignore_index=True)

nationalities = list(nationality_df.to_records(index = False))

with pd.option_context('display.max_rows',None,):
    print(nationality_df)

############################ language ##############################

languages = ["mandarin", "chinese", "spanish", "english", "hindi", "bengali", "portuguese", "russian", "japanese", "vietnamese", 
             "turkish", "marathi", "telugu", "punjabi", "korean", "tamil", "german", "french", "urdu", "arabic", 
             "javanese", "italian", "iranian persian", "gujarati", "hausa", "bhojpuri", "southern min"]

def determine_language(tag):
    highest_score = 0
    best_match = "NaN"
    
    # Check if any language is a standalone word in the tag
    for language in languages:
        if is_standalone_word(language, tag):
            return language
    
    # If no standalone match, proceed with fuzzy matching
    for language in languages:
        score = fuzz.ratio(tag, language)
        if score > highest_score:
            highest_score = score
            best_match = language
    
    return best_match if highest_score > 90 else "NaN"

# Group by cleaned_tag and apply the language determination function
unique_tags = lastfm_tags_df['cleaned_tag'].unique()
language_mapping = {tag: determine_language(tag) for tag in unique_tags}

# Map the gender back to the original DataFrame
lastfm_tags_df['language'] = lastfm_tags_df['cleaned_tag'].map(language_mapping)

# Display the DataFrame
lastfm_tags_df.head(50)
lastfm_tags_df[lastfm_tags_df['language'] != 'NaN']

######################################### religion/continents/other_geo ########################################

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

lastfm_tags_df.to_csv(r"C:\Users\resha\data\lastfm_diverse_tags_df.csv")  
lastfm_diverse_tags_df = lastfm_tags_df

