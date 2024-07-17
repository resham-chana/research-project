# install relevant packages

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
plt.yticks(fontsize=9)
ax.get_xaxis().set_major_formatter(
     mtick.FuncFormatter(lambda x, p: format(int(x), ',')))
labels = ax.bar_label(bar_container, fmt='{:,.0f}', color='white')
for label in labels:
    label.set_fontsize(9)
plt.savefig(r"C:\Users\resha\plots\popular_tags.png")
#plt.savefig(r"C:\Users\corc4\plots\popular_tags.png")

plt.show()


# cleaning tags

lastfm_tags_df['tag'] = lastfm_tags_df['tag'].astype(str)
lastfm_tags_df['cleaned_tag'] = lastfm_tags_df['tag'].str.replace(r'[^a-zA-Z0-9\s]', '', regex=True).str.strip().str.lower()
lastfm_tags_df = lastfm_tags_df[lastfm_tags_df['cleaned_tag'].str.len() > 1]
unique_tags = lastfm_tags_df['cleaned_tag'].unique()

########################## fuzzy grouping gender ##################################

# Define the words to match
female_terms = ['woman', 'female', 'female singer', 'female vocalist', 'female vocal', 'female voice']
male_terms = ['man', 'male','male singer', 'male vocalist']

def determine_gender(tag):
    # Initialize scores for female and male terms
    female_score = 0
    male_score = 0
    
    # Calculate fuzzy match scores for female terms using fuzz.ratio
    for female_term in female_terms:
        score = fuzz.ratio(tag, female_term)
        if score > 90:
            female_score = score
    
    # Calculate fuzzy match scores for male terms using fuzz.ratio or check if any male term is a substring of the tag
    for male_term in male_terms:
        score = fuzz.ratio(tag, male_term)
        if score > 90:
            male_score = score

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

############################ fuzzy grouping countries/languages/ethnicities/religion ##################################

# checking if the tags countain different languages:


geography_path = r"C:\Users\resha\data\geography_df.csv"
geography_df = pd.read_csv(geography_path)


# Function to clean and split nationality strings
def clean_and_split(strings):
    # Remove text after 'note'
    strings = re.sub(r'-', '', strings)
    strings = re.sub(r'[()]', ' ', strings) #brackets
    strings = re.sub(r'[0-9]+(?:\.[0-9]+)?|[%]', '', strings) #numbers, percentage
    strings = ' '.join(word for word in strings.split() if not word.islower()) # remove lower case words
    strings = strings.replace('~', '').replace('-', '').replace('<', '').replace('.','')
    # Split the text by commas, semicolons, 'or', and slashes
    split_strings = re.split(r'[;,/]| or ', strings)
    # Remove leading and trailing whitespace and filter out empty strings
    split_strings = [n.strip() for n in split_strings if n.strip()]
    # Remove any remaining unwanted words (if any)
    return split_strings

nationality_df = geography_df[['country','nationality']]
nationality_df.loc[:, 'nationality'] = nationality_df['nationality'].astype(str).apply(clean_and_split)
nationality_df = nationality_df.explode('nationality').reset_index(drop=True).drop_duplicates()
with pd.option_context('display.max_rows', None,'display.max_columns',None):
    print(nationality_df)

continent_df = geography_df[['continent']].drop_duplicates().drop([33,81,83,241,254])

ethnic_groups_df = geography_df[['country','ethnicity']]
ethnic_groups_df.loc[:, 'ethnicity'] = ethnic_groups_df['ethnicity'].astype(str).apply(clean_and_split)
ethnic_groups_df = ethnic_groups_df.explode('ethnicity').reset_index(drop=True).drop_duplicates("ethnicity")
with pd.option_context('display.max_rows', None,):
    print(ethnic_groups_df)

language_df = geography_df[['country','language']]
language_df.loc[:, 'language'] = language_df['language'].astype(str).apply(clean_and_split)
language_df = language_df.explode('language').reset_index(drop=True).drop_duplicates(subset="language")
with pd.option_context('display.max_rows', None,):
    print(language_df)

religion_df = geography_df[['country','religion']]
religion_df.loc[:, 'religion'] = religion_df['religion'].astype(str).apply(clean_and_split)
religion_df = religion_df.explode('religion').reset_index(drop=True).drop_duplicates(subset="religion")
with pd.option_context('display.max_rows', None,):
    print(religion_df)

# clean columns 

def clean_columns(dataframe, columns):
    dataframe = dataframe.dropna(subset=columns)    
    for column in columns:
        dataframe[column] = dataframe[column].str.replace(r'[^a-zA-Z0-9\s]', '', regex=True).str.strip().str.lower()
    return dataframe

nationality_df = clean_columns(nationality_df, ["country", "nationality"])
continent_df = clean_columns(continent_df, ["continent"])
ethnic_groups_df = clean_columns(ethnic_groups_df, ["ethnicity"])
language_df = clean_columns(language_df, ["language"])
religion_df = clean_columns(religion_df, ["religion"])


# new column if comma separated and / separated and 
# get rid of numbers and % and - and () and ; and make everything lowercase and the words 
# and/or/est/unspecified/other/mixed/including/unspecified/official/less/than/NaN/singular/plural/note/(s)
# make dashes / and - spaces
# new column based of commas

#nationality_df.to_csv(r"C:\Users\resha\data\nationality_df.csv")  
#continent_df.to_csv(r"C:\Users\resha\data\continent_df.csv")  
#ethnic_groups_df.to_csv(r"C:\Users\resha\data\ethnic_groups_df.csv")  
#religion_df.to_csv(r"C:\Users\resha\data\religion_df.csv")  

# Function to match tags with countries
def match_country(tag, nationality_df):
    # Special cases handling
    if 'uk' in tag:
        return 'united kingdom'
    if 'usa' in tag:
        return 'united states'
    
    for _, row in nationality_df.iterrows():
        country = row['country']
        nationality = row['nationality']
        
        # Fuzzy matching
        if fuzz.ratio(tag, country) > 90 or fuzz.ratio(tag, nationality) > 90:
            return country
        
        # Substring checks
        if nationality != 'nan' and nationality in tag:
            return country
        if country in tag:
            return country
    
    # Specific substring checks for common patterns
    if 'brit' in tag:
        return 'united kingdom'    
    return None

lastfm_tags_df['country'] = lastfm_tags_df['cleaned_tag'].apply(lambda x: match_country(x, nationality_df))

############################ language ##############################


# Function to match tags with countries
def match_country(tag, language_df):
    
    for _, row in language.iterrows():
        language = row['language']        
        # Fuzzy matching
        if fuzz.ratio(tag, language) > 90 or fuzz.ratio(tag, language) > 90:
            return language
        # Substring checks
        if language != 'nan' and language in tag:
            return language
    return None


for country in countries:
    country_match_bool = lastfm_tags_df['cleaned_tag'].str.contains(country)
    lastfm_tags_df.loc[country_match_bool, 'country'] = country

country_mismatch = lastfm_tags_df['country'].isna()
lastfm_tags_df.loc[country_mismatch, 'country'] = lastfm_tags_df.loc[country_mismatch, 'cleaned_tag'].apply(determine_country, countries=countries)

# Display the updated DataFrame
print(lastfm_tags_df[['cleaned_tag', 'country']])

contains_string = lastfm_tags_df['cleaned_tag'].str.contains(r"india|america|britain|pakistan", na=False).sum()


############################ clustering tags ##################################
import gensim.downloader as api
from sklearn.metrics.pairwise import cosine_similarity


glove_model = api.load("glove-wiki-gigaword-300")

glove_model["beautiful"]

lastfm_pivot_df

def get_tag_vector(tag, model):
    words = tag.split()
    vectors = [model[word] for word in words if word in model]
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        return None
    
tags = lastfm_pivot_df.iloc[:, 1:].stack().unique()  # Get unique tags, excluding NaNs
tag_vectors = {}
for tag in tags:
    if pd.notna(tag):
        vector = get_tag_vector(tag, glove_model)
        if vector is not None:
            tag_vectors[tag] = vector

def compute_song_vector(tags, tag_vectors):
    vectors = []
    for tag in tags:
        if tag in tag_vectors:
            vectors.append(tag_vectors[tag])
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(len(next(iter(tag_vectors.values()))))

lastfm_pivot_df['tags'] = lastfm_pivot_df.iloc[:, 1:].apply(lambda row: row.dropna().tolist(), axis=1)

lastfm_pivot_df['song_vector'] = lastfm_pivot_df['tags'].apply(lambda tags: compute_song_vector(tags, tag_vectors))

song_vectors = np.stack(lastfm_pivot_df['song_vector'].values)

similarity_matrix = cosine_similarity(song_vectors)

similarity_df = pd.DataFrame(similarity_matrix, index=lastfm_pivot_df['tid'], columns=lastfm_pivot_df['tid'])
print(similarity_df)

########################## getting genres ##################################

# get most popular tags:

tag_counts = lastfm_tags_df["tag"].value_counts()

for tag in tag_counts[:1000].index:
    print(tag)

tag_counts.to_csv('tags.csv', index=True)

lastfm_tags_df 

# unique tags:

len(lastfm_tags_df["tag"].unique()) # 505215 songs with 522366 unique tags

#"male", "female",""