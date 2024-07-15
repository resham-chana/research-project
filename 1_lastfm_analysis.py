# install relevant packages

import matplotlib
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from thefuzz import fuzz
from thefuzz import process
from wordcloud import WordCloud
import seaborn as sns


# read processed lastfm tag datasets
#lastfmpath = r"C:\Users\resha\data\lastfm_tags_df.csv"
#pivot_path = r"C:\Users\resha\data\lastfm_pivot_df.csv"
lastfmpath = r"C:\Users\corc4\data\lastfm_tags_df.csv"
pivot_path = r"C:\Users\corc4\data\lastfm_pivot_df.csv"
 
lastfm_tags_df = pd.read_csv(lastfmpath)
lastfm_pivot_df = pd.read_csv(pivot_path)

# visualise most common tags
tag_counts = lastfm_tags_df["tag"].value_counts()

# convert to DataFrame 
tag_counts_df = tag_counts.reset_index()
tag_counts_df.columns = ['tag', 'count']
tag_counts_df = tag_counts_df.head(20).sort_values("count", ascending=True)

plt.style.use("dark_background")
fig, ax = plt.subplots(figsize=(12,8))

bar_container = ax.barh("tag", "count", data=tag_counts_df, color="#4CAF50")

# Set plot title and labels
plt.title('Top 20 Tags by Count')
plt.xlabel('Count')
plt.ylabel('Tag')
plt.yticks(fontsize=9)
ax.get_xaxis().set_major_formatter(
     mtick.FuncFormatter(lambda x, p: format(int(x), ',')))
labels = ax.bar_label(bar_container, fmt='{:,.0f}', color='white')
for label in labels:
    label.set_fontsize(9)
#plt.savefig(r"C:\Users\resha\plots\popular_tags.png")
plt.savefig(r"C:\Users\corc4\plots\popular_tags.png")

plt.show()



########################## fuzzy grouping women ##################################

lastfm_tags_df['tag'] = lastfm_tags_df['tag'].astype(str)
lastfm_tags_df['cleaned_tag'] = lastfm_tags_df['tag'].str.replace(r'[^a-zA-Z0-9\s]', '', regex=True).str.strip().str.lower()
lastfm_tags_df = lastfm_tags_df[lastfm_tags_df['cleaned_tag'].str.len() > 1]
unique_tags = lastfm_tags_df['cleaned_tag'].unique()
 
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
    wordcloud.to_file(f"wordcloud_{gender}.png")
    plt.show()

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