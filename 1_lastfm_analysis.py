# install relevant packages

import matplotlib
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from thefuzz import fuzz
from thefuzz import process

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



########################## fuzzy grouping tags ##################################

lastfm_tags_df['tag'] = lastfm_tags_df['tag'].astype(str)
lastfm_tags_df['cleaned_tag'] = lastfm_tags_df['tag'].str.replace(r'[^a-zA-Z0-9\s]', '', regex=True).str.strip()
lastfm_tags_df = lastfm_tags_df[lastfm_tags_df['cleaned_tag'].str.len() > 1]
unique_tags = lastfm_tags_df['cleaned_tag'].unique()

# housekeeping 
len(unique_tags)
# Calculate tag popularity
tag_popularity = lastfm_tags_df['cleaned_tag'].value_counts()

lastfm_tags_subset_df = lastfm_tags_df.sample(frac=0.05, random_state=42)  # Use a fixed seed for reproducibility
unique_tags_subset = lastfm_tags_subset_df['cleaned_tag'].unique()

# Function to find the best match for a tag based on highest ratio and popularity
def find_best_match(tag, tag_dict, tag_popularity, threshold=80):
    # Extract top 5 potential matches
    candidates = process.extract(tag, tag_dict.keys(), scorer=fuzz.partial_ratio, limit=5)
    
    best_match = None
    highest_score = 0
    
    for candidate, score in candidates:
        if score >= threshold:
            # Choose the best match based on score and popularity
            if (score > highest_score) or (score == highest_score and tag_popularity[candidate] > tag_popularity.get(best_match, 0)):
                best_match = candidate
                highest_score = score
    
    if best_match:
        return tag_dict[best_match]
    return tag

# Create a dictionary to map cleaned tags to their groups
tag_mapping = {}
for tag in unique_tags_subset:
    if tag not in tag_mapping:
        tag_mapping[tag] = find_best_match(tag, tag_mapping, tag_popularity)

# Apply the tag mapping to the subset DataFrame
lastfm_tags_subset_df['tag_match'] = lastfm_tags_subset_df['cleaned_tag'].map(lambda x: tag_mapping.get(x, x))

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