import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()
import re
import ast
from scipy import stats
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from scipy.stats import uniform, randint
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, mean_absolute_error
import plotly.express as px
from sklearn.feature_selection import SelectKBest, f_regression, chi2, mutual_info_regression, f_classif


from scipy.stats import pearsonr, spearmanr
from category_encoders import TargetEncoder
from sklearn.preprocessing import LabelEncoder
import ast

pd.set_option('display.max_colwidth', None)

# In[2]:


df=pd.read_csv(r"C:\Users\yousef\Desktop\ML\SongPopularity.csv")

pd.set_option('display.max_columns', None)
df.head()

# **Hot100 Ranking Year:** The year in which the song achieved its ranking on the Billboard Hot 100 chart.
#
# **Hot100 Rank:** The specific ranking of the song on the Billboard Hot 100 chart during a particular year.
#
# **Acousticness:** tells us how much of a song is made with real instruments versus electronic ones: higher numbers mean more real instruments, while lower numbers mean more electronic sounds.
#
# **Danceability:** measures how easy it is to dance to a song: higher values mean it's easier to dance to, while lower values mean it might be harder to dance to.
#
# **Energy:** A measure of the song's intensity and activity, often associated with loudness and speed.
#
# **Instrumentalness:** Indicates the presence of vocals vs. instrumental elements in the song.
#
# **Liveness:** Reflects the likelihood of the song being performed live, based on audience noises and crowd sounds.
#
# **Speechiness:** Measures the presence of spoken words or speech-like elements in the song.
#
# **Tempo:** The speed or pace of the song, typically measured in beats per minute (BPM).
#
# **Valence:** Describes the musical positiveness conveyed by the song, such as happiness or cheerfulness.
#
# **Key:** The musical key or tonality of the song, which influences its mood and sound.
#
# **Time Signature:** Specifies the number of beats in each bar and the type of note that receives one beat, defining the song's rhythmic structure.

# # **Data Preprocessing**

# In[3]:


df.shape

# In[4]:


list(df.columns)

# In[5]:


df.info()

# All datatypes are correct

# In[6]:


df.describe().round(2)

# **Hot100 Ranking Year:** The ranking years range between 2010 and 2020
#
# **Hot100 Rank:** The average rank is 50.5 and There are no missing values.
#
# **Popularity:** The average popularity score is 66.4
#
# **Energy:** The minimum energy value is 0.0112, the maximum energy value is 0.996.
#
# **Instrumentalness:** The average instrumentalness value is 0.117.
#
# **Liveness:** The liveness value ranges from 0.0158 to 0.995.
#
# **Loudness:** The minimum loudness value is -14.874, the maximum loudness value is 5.787.
#
# **Acousticness:** The most acoustic songs tend to have an acousticness score of 0.8 or higher.
#
# **Danceability:** The distribution of danceability scores is approximately normal, with a slight skew to the right.
#
# **Key:** The distribution of key scores is approximately uniform.
#
# **Speechiness:** The distribution of speechiness scores is skewed to the right.
#
# **Tempo:** The fastest songs tend to have a tempo score of 120 or higher.
#
# **Valence:** The minimum valence score is 0.0337, the maximum valence score is 0.982.
#
# **Time Signature:** The average time signature is 4.0.
#
# **Mode:** There are two modes in the dataset: "Major" and "Minor"and the most common mode is "Major", which appears 61 times in the dataset.

# In[7]:


df.isnull().sum()

# In[8]:


df.duplicated().sum()

# No null or duplicated values

# ### Exploring unique values of some features

# In[9]:


df['Mode'].unique()

# In[10]:


df['Hot100 Ranking Year'].unique()

# In[11]:


df['Key'].unique()

# In[12]:


df['Time Signature'].unique()

# In[13]:


df['Artist Names'].value_counts()

# In the dataset, many artists have multiple songs, and the artist with the highest contribution is 'The Karaoke Channel' with 42 songs. Following them are 'Madonna' and 'Janet Jackson' with their respective song counts.

# In[14]:


album_counts = df['Album'].value_counts()
multiple_occurrences = album_counts[album_counts > 1]
print(multiple_occurrences)

# For albums, the most prolific is 'Greatest Hits' with 48 entries. This is followed by 'Super Hits' and '16 Most Requested Songs', each having their own respective counts.

# In[15]:


song_counts = df['Song'].value_counts()
multiple_occurrences = song_counts[song_counts > 1]
print(multiple_occurrences)

# **Many song names are duplicated!**

# In[16]:


df1 = df[df['Song'] == 'I Like It']
df1

# Song names are duplicated but not all links are

# In[17]:


df1['Spotify Link'].unique()

# In[18]:


link_counts = df['Spotify Link'].value_counts()
link_multiple_occurrences = link_counts[link_counts > 1]
print(link_multiple_occurrences)

# In[19]:


duplicate_links = df[df.duplicated(subset=['Spotify Link'], keep=False)]
duplicate_links

# ### If a link appears more than once but with different top 100 ranking years, it may not be an issue since the song could be featured in multiple significant song lists across different years. However, if the top 100 ranking years are identical for duplicated links, it presents a conflict for the model. This is uncommon because a song typically shouldn't have multiple rankings in the same year. Therefore, I will investigate rows where both the link and top 100 ranking year are duplicated. ###
#

# In[20]:


duplicate_links = df[df.duplicated(subset=['Spotify Link', 'Hot100 Ranking Year'], keep=False)]
duplicate_links.shape[0]

# In[21]:


duplicate_links

# Since there are only 12 rows where both the top 100 ranking year and the link are duplicated, I will drop these rows from the dataset.
#

# In[22]:


df.drop_duplicates(subset=['Spotify Link', 'Hot100 Ranking Year'], keep=False, inplace=True)

df.reset_index(drop=True, inplace=True)

# In[23]:


duplicate_links = df[df.duplicated(subset=['Spotify Link'], keep=False)]
duplicate_links.shape[0]

# In[24]:


link_counts = df['Spotify URI'].value_counts()
link_multiple_occurrences = link_counts[link_counts > 1]
print(link_multiple_occurrences)

# In[25]:


df['Hot100 Ranking Year'].value_counts()

# In[26]:


df['Album'].value_counts()

# The majority of songs appear to have rankings in the mid to high 90s, particularly in 2017 and 1974. This suggests that these years might have had a higher number of popular or significant songs. On the other hand, earlier years like 1955, 1950, and 1952 show lower rankings, indicating fewer popular songs or possibly a smaller dataset for those years.
#

# ### Handling Some Hidden Nulls

# Nulls in Artists' Genres are represented as "[]" in the dataset.

# In[27]:


df[df['Artist(s) Genres'].isin(["[]"])].head()

# In[28]:


df['Artist(s) Genres'].isin(["[]"]).sum()

# Filling null values in the Artists' Genres, which is categorical data, with the mode is a suitable approach. This will replace missing values with the most frequently occurring genre in the dataset.

# In[29]:


df['Artist(s) Genres'] = df['Artist(s) Genres'].apply(lambda x: np.nan if x == "[]" else x)
mode_value = df['Artist(s) Genres'].mode()[0]
df['Artist(s) Genres'].fillna(mode_value, inplace=True)

# In[30]:


print(df['Artist(s) Genres'].isin(["[]"]).sum())

# **Checking if there any other hidden Nulls**

# In[31]:


pattern = re.compile(r'[^a-zA-Z\s]')
df[df['Song'].str.contains(pattern, na=False)].head()

# In[32]:


pattern = r'^[^\w\s]+$'
HiddenNulls = df[df['Album'].str.match(pattern, na=False)]
HiddenNulls

# **There are hidden nulls in album column in form of "?" !!!**

# In[33]:


HiddenNulls.shape[0]

# In[34]:


rows_to_delete = df[df.isin(HiddenNulls.to_dict('list')).all(axis=1)].index

df.drop(rows_to_delete, inplace=True)

df.reset_index(drop=True, inplace=True)

# In[35]:


pattern = r'^[^\w\s]+$'
HiddenNulls = df[df['Album'].str.match(pattern, na=False)]
HiddenNulls.shape[0]

# I will convert the 'Album Release Date' column to only display the 'Year' since it is the most relevant information for our analysis.
#

# In[36]:


df['Year'] = df['Album Release Date'].apply(lambda x: x.split('/')[-1].split('-')[0])
df.head()

# A tempo of 0 is not possible for any song, so I will remove entries with a tempo value of 0 from the dataset.
#

# In[37]:


df[df['Tempo'] == 0].shape[0]

# In[38]:


'''
rows_to_delete = df[df['Tempo'] <= 0].index

df.drop(rows_to_delete, inplace=True)

df.reset_index(drop=True, inplace=True)
'''

# When an attempt was made to exclude instances where Tempo was equal to 0, it resulted in a decrease in accuracy. Consequently, these instances were retained in the dataset.

# # **EDA**

# In[39]:


cont = df[['Song Length(ms)', 'Acousticness', 'Danceability', 'Energy', 'Instrumentalness', 'Liveness', 'Loudness',
           'Speechiness', 'Tempo', 'Valence', 'Popularity']]
cat = df.drop(columns=cont.columns)

# **Boxplot**

# In[40]:


plt.figure(figsize=(15, 25))
for idx, i in enumerate(cont):
    plt.subplot(12, 2, idx + 1)
    sns.boxplot(x=i, data=df, palette="mako")
    plt.title(i, color='black', fontsize=15)
    plt.xlabel(i, size=12)
plt.tight_layout()
plt.show()

# There are outliers but we will handle them later.

# # **Histplots and Kdeplots**

# Check distributions

# In[41]:


fig, axs = plt.subplots(len(cont.columns), 2, figsize=(20, 60))

axs = axs.flatten()

for i, column in enumerate(cont.columns):
    sns.histplot(cont[column], bins=50, ax=axs[2 * i])
    axs[2 * i].set_title(f'Histogram of {column}')
    axs[2 * i].set_xlabel(column)
    axs[2 * i].set_ylabel('Frequency')

    sns.kdeplot(cont[column], ax=axs[2 * i + 1], fill=True)
    axs[2 * i + 1].set_title(f'KDE Plot of {column}')
    axs[2 * i + 1].set_xlabel(column)
    axs[2 * i + 1].set_ylabel('Density')
plt.tight_layout()
plt.show()


# The 'popularity' feature has over 250 zeros. We will further investigate to determine if these zeros represent null values or if they are
#

# In[42]:


def check_distribution(data):
    skewness = stats.skew(data)
    _, shapiro_p_value = stats.shapiro(data)

    if shapiro_p_value > 0.05:
        if skewness > 0:
            return "Right-skewed"
        elif skewness < 0:
            return "Left-skewed"
        else:
            return "Normally distributed"
    else:
        return "Not normally distributed"


results = cont.apply(check_distribution)

print(results)

# Some features, like 'danceability', appear to have a distribution close to normal. However, most of the other features seem to be right-skewed.
#

# In[43]:


# rows_to_delete = df[df['Tempo'] > 230].index
# df.drop(rows_to_delete, inplace=True)

# rows_to_delete = df[df['Loudness'] < -35].index
# df.drop(rows_to_delete, inplace=True)

# rows_to_delete = df[df['Speechiness'] > 0.8].index
# df.drop(rows_to_delete, inplace=True)

# df.reset_index(drop=True, inplace=True)


# In[44]:


list(cat.columns)

# **Countplot**

# Most artists have multiple songs in the dataset.
#

# In[45]:


duplicatedArtists = df[df['Artist Names'].duplicated()]
artist_counts = df['Artist Names'].value_counts()
top_30_artists = artist_counts.head(30)
plt.figure(figsize=(15, 10))
sns.countplot(y='Artist Names', data=df[df['Artist Names'].isin(top_30_artists.index)], order=top_30_artists.index,
              palette='mako')
plt.title('Count of Duplicated Artists (Top 30)')
plt.xlabel('Count')
plt.ylabel('Artist Names')
plt.show()

# Average popularity in these top 30 artists.

# In[46]:


top_30_artists = df['Artist Names'].value_counts().head(30).index

average_popularity = df[df['Artist Names'].isin(top_30_artists)].groupby('Artist Names')['Popularity'].mean()

plt.figure(figsize=(15, 10))
average_popularity.plot(kind='bar')
plt.title('Average Popularity of Songs by Top 30 Artists')
plt.xlabel('Artist Names')
plt.ylabel('Average Popularity')
plt.xticks(rotation=90)
plt.show()

# In[47]:


df[df['Artist Names'] == "['Ameritz Countdown Karaoke']"]

# The artists with the most popular songs are typically the most successful and famous in the industry.
#

# In[48]:


average_popularity = df.groupby('Artist Names')['Popularity'].mean()

top_30_artists = average_popularity.sort_values(ascending=False).head(30)

plt.figure(figsize=(15, 10))
top_30_artists.plot(kind='bar')
plt.title('Average Popularity of Songs by Top 30 Artists')
plt.xlabel('Artist Names')
plt.ylabel('Average Popularity')
plt.xticks(rotation=90)
plt.show()

# The albums with the most popular songs.
#

# In[49]:


average_popularity = df.groupby('Album')['Popularity'].mean()

top_30_albums = average_popularity.sort_values(ascending=False).head(30)

plt.figure(figsize=(15, 10))
top_30_albums.plot(kind='bar')
plt.title('Average Popularity of Songs from Top 30 Albums')
plt.xlabel('ALbums')
plt.ylabel('Average Popularity')
plt.xticks(rotation=90)
plt.show()

# In[50]:


pd.set_option('display.max_colwidth', None)
df[df['Popularity'] == 0]

# Most repeated years in the dataset

# In[51]:


plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='Year', palette='mako')
plt.title('Count Plot for Album Release Date')
plt.xlabel('Album Release Date')
plt.ylabel('Count')
plt.xticks(rotation=90)
plt.xticks(fontsize=8)
plt.yticks(fontsize=8)
plt.show()

# In[52]:


plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='Hot100 Ranking Year', palette='mako')
plt.title('Count Plot for Album Release Date')
plt.xlabel('Album Release Date')
plt.ylabel('Count')
plt.xticks(rotation=90)
plt.xticks(fontsize=8)
plt.yticks(fontsize=8)
plt.show()

# In[53]:


df['Year'] = df['Year'].astype(int)

# Albums with most columns

# In[54]:


album_counts = df['Album'].value_counts().reset_index()
album_counts.columns = ['Album', 'Count']
top_30_albums = album_counts.head(50)

plt.figure(figsize=(15, 8))
sns.barplot(data=top_30_albums, x='Album', y='Count', palette='mako')
plt.title('Top 30 Albums by Number of Songs')
plt.xlabel('Album')
plt.ylabel('Number of Songs')
plt.xticks(rotation=90)
plt.xticks(fontsize=8)
plt.yticks(fontsize=8)
plt.show()

# **Scatterplots**

# In[55]:


plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, y='Year', x='Popularity', palette='mako')
plt.title('Scatter Plot of Popularity vs Year')
plt.ylabel('Year')
plt.xlabel('Popularity')
plt.grid(True)
plt.show()

# In[56]:


plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, y='Hot100 Ranking Year', hue='Hot100 Rank', x='Popularity', palette='mako')
plt.title('Hot100 Ranking Year vs Popularity')
plt.xlabel('Popularity')
plt.ylabel('Hot 100 Ranking Year')
plt.grid(True)
plt.show()

# In[57]:


plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, y='Valence', x='Popularity', palette='mako')
plt.title('Popularity vs Valence')
plt.xlabel('Popularity')
plt.ylabel('Valence')
plt.grid(True)
plt.show()

# no direct relation betweeen valence and popularity

# In[58]:


plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, y='Danceability', x='Popularity', palette='mako')
plt.title('Popularity vs Danceability')
plt.xlabel('Popularity')
plt.ylabel('Danceability')
plt.grid(True)
plt.show()

# no direct relation betweeen valence and popularity

# In[59]:


plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, y='Tempo', x='Popularity', palette='mako')
plt.title('Popularity vs Tempo')
plt.xlabel('Popularity')
plt.ylabel('Tempo')
plt.grid(True)
plt.show()

# In[60]:


plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, y='Loudness', x='Popularity', palette='mako')
plt.title('Popularity vs Loudness')
plt.xlabel('Popularity')
plt.ylabel('Loudness')
plt.grid(True)
plt.show()

# In[61]:


plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, y='Danceability', x='Valence', hue='Popularity', palette='mako')
plt.title('Scatter Plot of Popularity vs Year')
plt.xlabel('Valence')
plt.ylabel('Danceability')
plt.grid(True)
plt.show()

# In[62]:


plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, y='Danceability', x='Popularity', palette='mako')
plt.title('Popularity vs Danceability')
plt.xlabel('Popularity')
plt.ylabel('Danceability')
plt.grid(True)
plt.show()

# In[63]:


plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, y='Energy', x='Loudness', hue='Popularity', palette='mako')
plt.title('Energy vs Loudness')
plt.xlabel('Loudness')
plt.ylabel('Energy')
plt.grid(True)
plt.show()

# In[64]:


plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, y='Acousticness', x='Energy', hue='Popularity', palette='mako')
plt.title('Acousticness vs Energy')
plt.xlabel('Energy')
plt.ylabel('Acousticness')
plt.grid(True)
plt.show()

# In[65]:


df['minutes_length'] = df['Song Length(ms)'].apply(lambda x: x / 60000)
df.head()

# In[66]:


plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, y='minutes_length', x='Popularity', hue='Popularity', palette='mako')
plt.title('Scatter Plot of Popularity vs Length')
plt.ylabel('Length')
plt.xlabel('Popularity')
plt.grid(True)
plt.show()

# **Pie Charts**

# In[67]:


original_speechiness_values = df['Speechiness'].copy()


# In[68]:


def type_of_Song(x):
    if x >= 0.0 and x < 0.1:
        return "very low"
    elif x >= 0.1 and x < 0.3:
        return "low"
    elif x >= 0.3 and x < 0.5:
        return "medium"
    elif x >= 0.5 and x < 0.7:
        return "high"
    else:
        return "very high"


# In[69]:


df['Speechiness'] = df['Speechiness'].apply(type_of_Song)

# In[70]:


n_songs_per_category = df.groupby('Speechiness').size()
fig = px.pie(df, names=n_songs_per_category.index, values=n_songs_per_category.values)
fig.update_layout(title='type of songs')
fig.show()

# In[71]:


df['Speechiness'] = original_speechiness_values

# In[72]:


Tempo_original_value = df['Tempo'].copy()


# In[73]:


def classify_tempo(bpm):
    if bpm < 90:
        return "Slow"
    elif 90 <= bpm <= 130:
        return "Moderate"
    else:
        return "Fast"


# In[74]:


df['Tempo'] = df['Tempo'].apply(classify_tempo)

# In[75]:


n_songs_per_category = df.groupby('Tempo').size()

fig = px.pie(names=n_songs_per_category.index,
             values=n_songs_per_category.values,
             title='Types of Tempo')

fig.update_layout(title='Types of Tempo')
fig.show()

# In[76]:


df['Tempo'] = Tempo_original_value

# In[77]:


valence_original_value = df['Valence'].copy()


# In[78]:


def valence_type(x):
    if x >= 0.0 and x < 0.5:
        return "Happy|Positive"
    elif x >= 0.5 and x < 1:
        return "Sad|Negative"


df['Valence'] = df['Valence'].apply(valence_type)

# In[79]:


n_songs_per_category = df.groupby('Valence').size()
fig = px.pie(df, names=n_songs_per_category.index, values=n_songs_per_category.values)
fig.update_layout(title='types of valence (Happy or sad)')

fig.show()

# In[80]:


df['Valence'] = valence_original_value

# **How songs popularity increases over time**

# In[81]:


df_sorted = df.sort_values(by='Year')

bins = range(1900, 2030, 10)
labels = [f"{i}-{i + 9}" for i in range(1900, 2020, 10)]

df_sorted['Year Group'] = pd.cut(df_sorted['Year'], bins=bins, labels=labels, right=False)

avg_popularity_by_year = df_sorted.groupby('Year Group')['Popularity'].mean()

plt.figure(figsize=(15, 5))
plt.plot(avg_popularity_by_year.index, avg_popularity_by_year.values, marker='o', linestyle='-')
plt.title('Average Popularity Over 10-Year Intervals (1900-2019)')
plt.xlabel('Year Interval')
plt.ylabel('Average Popularity')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()

# Higher popularities tend to be in more recent years. As we look further back in time, the popularities of songs generally decrease.
#

# ## Insights from EDA Visualizations
#
# **Hot Ranking Year & Popularity**: There is a direct proportional relationship between the hot ranking year and popularity rank.
#
#
# **Valence & Popularity**: There is no direct relationship between valence and popularity, loudness, or tempo.
#
#
# **Valence & Danceability**: Valence and danceability are somewhat directly proportional. Songs with higher danceability tend to have higher popularity.
#
#
# **Energy, Loudness & Popularity**: Energy and loudness are directly proportional to each other. Higher loudness and energy levels correlate with higher popularity.
#
#
# **Acousticness & Energy**: Acousticness and energy show an inverse relationship. Popularity tends to be higher with lower acousticness.
#
#
# **Song Length & Popularity**: There is no direct proportional relationship between song length and popularity. However, songs with extremely long lengths do not have high popularity.
#
#
# **Speechiness**: Most of the dataset has very low speechiness.
#
#
# **Tempo & Speed**: Over 54% of the data indicates songs that are neither too fast nor too slow. Fast songs with high tempo are twice as common as slow songs.
#
#
# **Mood & Popularity**: More than 64% of the dataset comprises sad songs. Interestingly, sad songs seem to have higher popularity.
#

# # **Feature Engineering**

# In[82]:


df.head()

# In[83]:


df.drop(['Album Release Date', 'minutes_length'], axis=1, inplace=True)

df.reset_index(drop=True, inplace=True)

# In[84]:


df = df[['Song', 'Album', 'Year', 'Artist Names', 'Artist(s) Genres',
         'Hot100 Ranking Year', 'Hot100 Rank', 'Song Length(ms)', 'Spotify Link',
         'Song Image', 'Spotify URI', 'Acousticness',
         'Danceability', 'Energy', 'Instrumentalness', 'Liveness', 'Loudness',
         'Speechiness', 'Tempo', 'Valence', 'Key', 'Mode', 'Time Signature', 'Popularity']]

# ## Encoding

# ### **Since the 'artist name' and 'artist genres' columns contain lists of strings, I will split each list into multiple rows, with each element in its own row. I will perform this transformation for both the 'artist name' and 'genre' columns.**

# In[85]:


df['Artist Names'] = df['Artist Names'].apply(ast.literal_eval)
df['Artist(s) Genres'] = df['Artist(s) Genres'].apply(ast.literal_eval)

df_exploded = df.explode('Artist Names')
df_exploded = df_exploded.explode('Artist(s) Genres')

df_exploded.info()

# In[86]:


df_exploded

# ### After splitting the 'artist name' and 'artist genres' columns into individual rows, I applied both target and label encoding to these features. Subsequently, I combined the encoded values back into a single cell for lists with more than one string.
#

# Target Encoding for Artist Genres

# In[87]:


encoder = TargetEncoder(cols=['Artist(s) Genres'])
encoder.fit(df_exploded, df_exploded['Popularity'])
df_encoded = encoder.transform(df_exploded)

# Label Encoding for Artist Genres

# In[88]:


le = LabelEncoder()
df_encoded['Artist Names'] = le.fit_transform(df_encoded['Artist Names'])

# In[89]:


df_encoded


# In[90]:


def aggregate_rows(group):
    sum_artists = sum(group['Artist Names'].unique())
    sum_genres = sum(group['Artist(s) Genres'].unique())

    return pd.Series({
        'Artist Names Encoded': sum_artists,
        'Artist(s) Genres Encoded': sum_genres
    })


aggregated_df = df_encoded.groupby(df_encoded.index).apply(aggregate_rows)

aggregated_df = aggregated_df.reset_index(drop=True)

aggregated_df.info()

# In[91]:


aggregated_df

# In[92]:


original_indices = set(df.index)
aggregated_indices = set(aggregated_df.index)

missing_indices = original_indices - aggregated_indices
extra_indices = aggregated_indices - original_indices

print("Missing indices:", missing_indices)
print("Extra indices:", extra_indices)

# In[93]:


df = pd.concat([aggregated_df, df], axis=1)

# In[94]:


df

# In[95]:


columns_to_drop = ['Artist Names', 'Artist(s) Genres', 'Song', 'Album', 'Spotify Link', 'Song Image', 'Spotify URI']

df.drop(columns_to_drop, axis=1, inplace=True)
df.reset_index(drop=True, inplace=True)

# In[96]:


column_names = {
    'Artist Names Encoded': 'Artist Names',
    'Artist(s) Genres Encoded': 'Artist(s) Genres'
}

df = df.rename(columns=column_names)

# In[97]:


df.head()

# In[98]:


'''
df = df[df['Popularity'] != 0]
df = df.reset_index(drop=True)
'''

# ## Creating new features

# The 'Hype' feature is calculated as the sum of 'Loudness' and 'Energy'.
#
# The 'Happiness' feature represents the sum of 'Danceability' and 'Valence'.

# In[99]:


df['Hype'] = df['Loudness'] + df['Energy']
df['Happiness'] = df['Danceability'] + df['Valence']

# ## Data Splitting

# In[100]:


X_feature = df.drop(['Popularity'], axis=1)
Y_feature = df['Popularity']
X_train, X_test, y_train, y_test = train_test_split(X_feature, Y_feature, test_size=0.20, shuffle=True, random_state=10)

# ## Transformations ##

# In[101]:


fig, axs = plt.subplots(len(df.columns), 2, figsize=(20, 60))

axs = axs.flatten()

for i, column in enumerate(df.columns):
    sns.histplot(df[column], bins=50, ax=axs[2 * i])
    axs[2 * i].set_title(f'Histogram of {column}')
    axs[2 * i].set_xlabel(column)
    axs[2 * i].set_ylabel('Frequency')

    sns.kdeplot(df[column], ax=axs[2 * i + 1], fill=True)
    axs[2 * i + 1].set_title(f'KDE Plot of {column}')
    axs[2 * i + 1].set_xlabel(column)
    axs[2 * i + 1].set_ylabel('Density')
plt.tight_layout()
plt.show()

# ## To address the skewed distributions, I will apply both a log transformation and a square root transformation to the relevant features.
#

# In[102]:


offset = 1e-10

X_train['Liveness_log'] = np.log(X_train['Liveness'] + offset)

sns.kdeplot(X_train['Liveness_log'], fill=True)
plt.title('Kernel Density Estimation Plot for x_train')
plt.xlabel('Values')
plt.ylabel('Density')
plt.show()

# In[103]:


X_test['Liveness_log'] = np.log(X_test['Liveness'] + offset)

sns.kdeplot(X_test['Liveness_log'], fill=True)
plt.title('Kernel Density Estimation Plot for x_test')
plt.xlabel('Values')
plt.ylabel('Density')
plt.show()

# In[104]:


offset = 1e-10

X_train['Acousticness_sqrt'] = np.sqrt(X_train['Acousticness'] + offset)

sns.kdeplot(X_train['Acousticness_sqrt'], fill=True)
plt.title('Kernel Density Estimation Plot for x_train')
plt.xlabel('Values')
plt.ylabel('Density')
plt.show()

# In[105]:


X_test['Acousticness_sqrt'] = np.sqrt(X_test['Acousticness'] + offset)

sns.kdeplot(X_test['Acousticness_sqrt'], fill=True)
plt.title('Kernel Density Estimation Plot for x_test')
plt.xlabel('Values')
plt.ylabel('Density')
plt.show()

# In[106]:


X_train['Speechiness_sqrt'] = np.sqrt(X_train['Speechiness'])

sns.kdeplot(X_train['Speechiness_sqrt'], fill=True, warn_singular=False)
plt.title('Kernel Density Estimation Plot for x_train')
plt.xlabel('Values')
plt.ylabel('Density')
plt.show()

# In[107]:


X_test['Speechiness_sqrt'] = np.sqrt(X_test['Speechiness'])
sns.kdeplot(X_test['Speechiness_sqrt'], fill=True, warn_singular=False)
plt.title('Kernel Density Estimation Plot for x_test')
plt.xlabel('Values')
plt.ylabel('Density')
plt.show()

# In[108]:


X_train.head()

# In[109]:


columns_to_drop = ['Liveness', 'Acousticness', 'Speechiness']

X_train.drop(columns_to_drop, axis=1, inplace=True)
X_train.reset_index(drop=True, inplace=True)
X_train.head()

# In[110]:


column_names = {
    'Liveness_log': 'Liveness',
    'Acousticness_sqrt': 'Acousticness',
    'Speechiness_sqrt': 'Speechiness'
}

X_train = X_train.rename(columns=column_names)
X_train.head()

# In[111]:


X_test.head()

# In[112]:


columns_to_drop = ['Liveness', 'Acousticness', 'Speechiness']

X_test.drop(columns_to_drop, axis=1, inplace=True)
X_test.reset_index(drop=True, inplace=True)
X_test.head()

# In[113]:


column_names = {
    'Liveness_log': 'Liveness',
    'Acousticness_sqrt': 'Acousticness',
    'Speechiness_sqrt': 'Speechiness'
}

X_test = X_test.rename(columns=column_names)
X_test.head()

# ## Scaling

# ### Normalization ###

# During the scaling process, the dataset was initially partitioned to avoid data leakage. Min-Max scaling was employed as it useful when data is not Gaussian, while Standard scaling was not utilized. Standard scaling is typically applied to normally distributed data, making Min-Max scaling the preferred choice for this dataset.

# In[114]:


scaler = MinMaxScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns)

y_scaler = MinMaxScaler()

y_train_scaled = y_scaler.fit_transform(y_train.values.reshape(-1, 1))
y_test_scaled = y_scaler.transform(y_test.values.reshape(-1, 1))

y_train_scaled_df = pd.DataFrame(y_train_scaled, columns=['Scaled Popularity'])
y_test_scaled_df = pd.DataFrame(y_test_scaled, columns=['Scaled Popularity'])

plt.hist(X_train_scaled_df.iloc[:, 0])
plt.title('Histogram of First Scaled Feature in Training Set')
plt.xlabel('Scaled Values')
plt.ylabel('Frequency')
plt.show()

# In[115]:


X_train_scaled_df.head()

# In[116]:


X_test_scaled_df.head()

# In[117]:


X_train = X_train_scaled_df
X_test = X_test_scaled_df
y_train = y_train_scaled_df
y_test = y_test_scaled_df

# # Feature Selection

# In[118]:


df[df['Year'] > df['Hot100 Ranking Year']]

# In[119]:


df[df['Year'] > df['Hot100 Ranking Year']].shape[0]

# There are over 1760 rows, accounting for more than 28% of the data, with a 'rank year' of 0 even before the song's release year. This suggests that the 'rank year' column is largely inaccurate.

# In[120]:


df['Time Signature'].value_counts() / df.shape[0]

# In[121]:


sns.violinplot(X_train['Time Signature'], fill=True)
plt.title('Kernel Density Estimation Plot for x_train')
plt.xlabel('Values')
plt.ylabel('Density')
plt.show()

# In[122]:


df['Instrumentalness'].value_counts() / df.shape[0]

# In[123]:


sns.violinplot(X_train['Instrumentalness'], fill=True)
plt.title('Kernel Density Estimation Plot for x_train')
plt.xlabel('Values')
plt.ylabel('Density')
plt.show()

# Instrumentalness and Time Signature will not be useddue to their negligible variance, indicating that they would not significantly influence the model. Specifically, over 99% of the Time Signature values were '4', and nearly half of the Instrumentalness values were zeros.

# ##  Correlation heatmap

# Since the correlation matrix is most informative with continuous data, I will exclude categorical data when generating the correlation heatmap.
#

# In[124]:


excluded_columns = ['Artist Names', 'Artist(s) Genres', 'Year', 'Hot100 Rank', 'Key', 'Mode', 'Time Signature']

X_correlation = df.drop(excluded_columns, axis=1)
y = df['Popularity']

numerical_features = X_correlation.select_dtypes(include=['int64', 'float64'])
categorical_features = X_correlation.select_dtypes(exclude=['int64', 'float64'])

numerical_correlations = numerical_features.corrwith(y)

sorted_correlations = numerical_correlations.abs().sort_values(ascending=False)
top_n_features = 18
top_features = sorted_correlations.index[:top_n_features]

top_feature_corr = X_correlation[top_features].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(top_feature_corr, annot=True, cmap='mako', fmt=".2f")
plt.title(f"Correlation Heatmap of Top {top_n_features} Features with Popularity")
plt.xlabel("Features")
plt.ylabel("Features")
plt.show()

# ##  Filter methods (SelectKBest)

# ### I will use SelectKBest to select some features and reduce the dimensionality of the model. First, I'll split the features into continuous and categorical. For the categorical features, I will select a subset using ANOVA, as the input is categorical and the output is numerical. For the continuous features, I will select a subset using both Pearson and Spearman correlation methods, as these are appropriate when both the input and output are numerical.
#

# In[125]:


cat_columns = ['Artist Names', 'Artist(s) Genres', 'Year', 'Hot100 Rank', 'Key', 'Mode', 'Time Signature',
               'Hot100 Ranking Year']

X_train_cat = X_train[cat_columns]

best_cat_features = SelectKBest(score_func=f_regression, k='all')

fit = best_cat_features.fit(X_train_cat, y_train)

df_scores = pd.DataFrame(fit.scores_, columns=['Score'])
df_columns = pd.DataFrame(X_train_cat.columns, columns=['Feature'])

featureScores = pd.concat([df_columns, df_scores], axis=1)

print(featureScores.nlargest(33, 'Score'))


# In[126]:


def spearman_corr(X, y):
    scores = []
    dummy_pvalues = [None] * X.shape[1]
    for feature in X.T:
        corr, _ = spearmanr(feature, y)
        scores.append(abs(corr))
    return scores, dummy_pvalues


# In[127]:


cont_columns = df.drop(
    ['Artist Names', 'Artist(s) Genres', 'Year', 'Hot100 Rank', 'Key', 'Mode', 'Time Signature', 'Popularity',
     'Hot100 Ranking Year'], axis=1).columns

X_train_cont = X_train[cont_columns]

best_cont_features = SelectKBest(score_func=spearman_corr, k='all')

fit = best_cont_features.fit(X_train_cont, y_train)

df_scores = pd.DataFrame(fit.scores_, columns=['Score'])
df_columns = pd.DataFrame(X_train_cont.columns, columns=['Feature'])

featureScores = pd.concat([df_columns, df_scores], axis=1)

print(featureScores.nlargest(33, 'Score'))

# In[128]:


selected_features = ['Acousticness', 'Artist(s) Genres', 'Hot100 Ranking Year', 'Mode', 'Year', 'Hot100 Rank',
                     'Song Length(ms)', 'Hype']

X_train_selected = X_train[selected_features]
X_test_selected = X_test[selected_features]
X_train = X_train_selected
X_test = X_test_selected

# In[129]:


X_train.head()

# In[130]:


X_test.head()

# # Modeling & Hyperparameter tuning

# **Hyperparameter Tuning**
#
# Hyperparameter tuning was conducted using grid search, which systematically combined all possible combinations of hyperparameters provided. This approach aimed to identify the optimal hyperparameter values that yielded the best-performing models in terms of accuracy. The tuning process significantly enhanced the accuracy of the models by optimizing the hyperparameters.
#
# **Cross-Validation**
#
# Cross-validation was employed to assess the generalization capability of the models and mitigate overfitting. This provided a more reliable estimate of a model's performance by averaging the evaluation results across multiple validation sets, thereby offering a more robust evaluation metric compared to a single train-test split.

# # Linear Regression

# In[131]:


model = LinearRegression()
cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
mean_mse = -np.mean(cv_scores)

print("Mean Cross-Validated MSE:", mean_mse)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse_test = mean_squared_error(y_test, y_pred)
RMSE = mean_squared_error(y_test, y_pred, squared=False)
MAE = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

y_pred_train = model.predict(X_train)
mse_train = mean_squared_error(y_train, y_pred_train)

print("\nTraining Set Mean Squared Error (MSE):", mse_train)
print("Test Set Metrics:")
print("Mean Squared Error (MSE):", mse_test)
print("Root Mean Squared Error:", RMSE)
print("Mean Absolute Error:", MAE)
print("R-squared Score:", r2)

for i, feature in enumerate(selected_features):
    model.fit(X_train[[feature]], y_train)

    coefficients = model.coef_
    intercept = model.intercept_

    y_pred = model.predict(X_train[[feature]])

    plt.figure(figsize=(8, 6))
    plt.scatter(X_train[[feature]], y_train, color='blue', label='Actual')
    plt.plot(X_train[[feature]], y_pred, color='red', label='Predicted')
    plt.xlabel(feature)
    plt.ylabel('Popularity')
    plt.title(f'Linear Regression Line for {feature}')
    plt.legend()
    plt.show()

# # Support Vector Machine (SVR) Regression

# In[141]:


param_grid_svm = {
    'C': [0.1, 1, 10],
    'gamma': [0.1, 1, 10],
    'kernel': ['linear', 'rbf']
}

svm_model = SVR()

grid_search_svm = GridSearchCV(estimator=svm_model, param_grid=param_grid_svm, cv=5, scoring='neg_mean_squared_error',
                               verbose=0)
grid_search_svm.fit(X_train_scaled, y_train)

best_params_svm = grid_search_svm.best_params_
best_score_svm = -grid_search_svm.best_score_

svm_model_best = SVR(**best_params_svm)
svm_model_best.fit(X_train_scaled, y_train)

y_pred_svm = svm_model_best.predict(X_test_scaled)

mse_svm = mean_squared_error(y_test, y_pred_svm)
RMSE = mean_squared_error(y_test, y_pred_svm, squared=False)
MAE = mean_absolute_error(y_test, y_pred_svm)
r2_svm = r2_score(y_test, y_pred_svm)

print("Support Vector Machine (SVM) Regression:")
print(f"Best hyperparameters: {best_params_svm}")
print(f"Mean cross-validation error: {best_score_svm}")
print(f"Mean squared error: {mse_svm}")
print(f"Root mean squared error: {RMSE}")
print("Mean Absolute Error:", MAE)
print(f"R-squared score: {r2_svm}")

# # Decision Tree Regressor

# In[133]:


param_grid_dt = {
    'max_depth': [None, 5, 10, 15],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

dt_model = DecisionTreeRegressor()

grid_search_dt = GridSearchCV(estimator=dt_model, param_grid=param_grid_dt, cv=5, scoring='neg_mean_squared_error')
grid_search_dt.fit(X_train, y_train)

best_params_dt = grid_search_dt.best_params_
best_score_dt = -grid_search_dt.best_score_

dt_model_best = DecisionTreeRegressor(**best_params_dt)
dt_model_best.fit(X_train, y_train)

y_pred_dt = dt_model_best.predict(X_test)

mse_dt = mean_squared_error(y_test, y_pred_dt)
RMSE = mean_squared_error(y_test, y_pred_dt, squared=False)
MAE = mean_absolute_error(y_test, y_pred_dt)
r2_dt = r2_score(y_test, y_pred_dt)

print("\nDecision Tree Regression:")
print(f"Best hyperparameters: {best_params_dt}")
print(f"Mean cross-validation error: {best_score_dt}")
print(f"Mean squared error: {mse_dt}")
print("Root Mean Squared Error:", RMSE)
print("Mean Absolute Error:", MAE)
print(f"R-squared score: {r2_dt}")

# # Polynomial Regression

# ## Degree 3

# In[143]:


degree = 3
poly = PolynomialFeatures(degree=degree)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

model = LinearRegression()
model.fit(X_train_poly, y_train)

y_pred = model.predict(X_test_poly)

cv_scores = cross_val_score(model, X_train_poly, y_train, cv=5, scoring='neg_mean_squared_error')
mean_mse_cv = -np.mean(cv_scores)

print(f"Polynomial Regression (degree={degree}):")
print(f"Mean squared error (CV): {mean_mse_cv}")

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
RMSE = mean_squared_error(y_test, y_pred, squared=False)
MAE = mean_absolute_error(y_test, y_pred)
print(f"Mean squared error: {mse}")
print("Root Mean Squared Error:", RMSE)
print("Mean Absolute Error:", MAE)
print(f"R-squared score: {r2}")

for i, feature in enumerate(selected_features):
    plt.figure(figsize=(10, 5))

    plt.scatter(X_train[feature], y_train, color='blue', label='Actual (Training Data)')

    sorted_indices = X_test[feature].argsort()
    plt.plot(X_test[feature][sorted_indices], y_pred[sorted_indices], color='red', label='Predicted')

    plt.title(f'Scatter Plot and Polynomial Regression Line for {feature}')
    plt.xlabel(feature)
    plt.ylabel('Target (y)')
    plt.legend()
    plt.show()

# ## Degree 2

# In[144]:


degree = 2
poly = PolynomialFeatures(degree=degree)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

model = LinearRegression()
model.fit(X_train_poly, y_train)

y_pred = model.predict(X_test_poly)

cv_scores = cross_val_score(model, X_train_poly, y_train, cv=5, scoring='neg_mean_squared_error')
mean_mse_cv = -np.mean(cv_scores)

print(f"Polynomial Regression (degree={degree}):")
print(f"Mean squared error (CV): {mean_mse_cv}")

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
RMSE = mean_squared_error(y_test, y_pred, squared=False)
MAE = mean_absolute_error(y_test, y_pred)
print(f"Mean squared error: {mse}")
print("Root Mean Squared Error:", RMSE)
print("Mean Absolute Error:", MAE)
print(f"R-squared score: {r2}")

# # Random Forest Regressor

# In[136]:


param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

rf_regressor = RandomForestRegressor(random_state=42)

grid_search = GridSearchCV(estimator=rf_regressor, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error',
                           verbose=0)
grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
best_score = -grid_search.best_score_
# best_params = {'max_depth': None,'min_samples_leaf': 2,'min_samples_split': 5,'n_estimators': 150}
rf_best = RandomForestRegressor(**best_params, random_state=42)
rf_best.fit(X_train, y_train)

y_pred = rf_best.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
RMSE = mean_squared_error(y_test, y_pred, squared=False)
MAE = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Random Forest regression:")
print("Best hyperparameters:", best_params)
print(f"Mean cross-validation error: {best_score}")
print("Mean squared error:", mse)
print("Root Mean Squared Error:", RMSE)
print("Mean Absolute Error:", MAE)
print("R-squared score:", r2)

for i, feature in enumerate(selected_features):
    plt.figure(figsize=(10, 5))
    plt.scatter(X_train[feature], y_train, color='blue', label='Actual (Training Data)')

    sorted_indices = X_test[feature].argsort()

    plt.plot(X_test[feature][sorted_indices], y_pred[sorted_indices], color='red', label='Predicted')

    plt.title(f'Scatter Plot and Random Forest Regression Line for {feature}')
    plt.xlabel(feature)
    plt.ylabel('Target (y)')
    plt.legend()
    plt.show()

# # XGBoost Regression

# In[137]:


# !pip install xgboost -q


# In[138]:


param_grid = {
    'max_depth': [3, 4, 5],
    'learning_rate': [0.1, 0.01, 0.001],
    'n_estimators': [100, 200, 300]
}

xgb_regressor = xgb.XGBRegressor()

grid_search = GridSearchCV(estimator=xgb_regressor, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error',
                           verbose=0)
grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
best_score = -grid_search.best_score_
best_params = {'learning_rate': 0.1, 'max_depth': 5, 'n_estimators': 100}

best_xgb_regressor = xgb.XGBRegressor(**best_params)
best_xgb_regressor.fit(X_train, y_train)

y_pred = best_xgb_regressor.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
RMSE = mean_squared_error(y_test, y_pred, squared=False)
MAE = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("XGBoost regression:")
print("Best hyperparameters:", best_params)
print(f"Mean cross-validation error: {best_score}")
print("Mean squared error:", mse)
print("Root Mean Squared Error:", RMSE)
print("Mean Absolute Error:", MAE)
print("R-squared score:", r2)

# # Lasso Regression

# In[145]:


alphas = [0.001, 0.01, 0.1, 1.0, 10.0]

lasso = Lasso()

grid_search = GridSearchCV(estimator=lasso, param_grid={'alpha': alphas}, cv=5, scoring='neg_mean_squared_error',
                           verbose=1)

grid_search.fit(X_train, y_train)

best_alpha = grid_search.best_params_['alpha']
best_mse = -grid_search.best_score_
LR = grid_search.best_estimator_

print("Best alpha:", best_alpha)
print("Best MSE:", best_mse)

test_mse = -grid_search.score(X_test, y_test)
print("Test MSE:", test_mse)
kf = KFold(n_splits=10, shuffle=True, random_state=1)

scores = cross_val_score(LR, X_train_scaled, y_train, scoring='neg_mean_squared_error', cv=kf)
mean_score = -np.mean(scores)
std_score = np.std(scores)
print("Mean Score:", mean_score)
print("Standard Deviation Score:", std_score)

LR.fit(X_train_scaled, y_train)
y_pred = LR.predict(X_test_scaled)

MSE = mean_squared_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error = ", MSE)
print("Root Mean Squared Error = ", rmse)
print("Mean absolute Error = ", mae)
print("R2_score = ", r2)

for i, feature in enumerate(selected_features):
    plt.figure(figsize=(10, 5))
    plt.scatter(X_train[feature], y_train, color='blue', label='Actual (Training Data)')

    sorted_indices = X_test[feature].argsort()

    plt.plot(X_test[feature][sorted_indices], y_pred[sorted_indices], color='red', label='Predicted')

    plt.title(f'Scatter Plot and Random Forest Regression Line for {feature}')
    plt.xlabel(feature)
    plt.ylabel('Target (y)')
    plt.legend()
    plt.show()

# # Ridge Regression

# In[140]:


alphas = [0.001, 0.01, 0.1, 1.0, 10.0]

ridge = Ridge()

grid_search = GridSearchCV(estimator=ridge, param_grid={'alpha': alphas}, cv=5, scoring='neg_mean_squared_error',
                           verbose=1)

grid_search.fit(X_train, y_train)

best_alpha = grid_search.best_params_['alpha']
best_mse = -grid_search.best_score_
RR = grid_search.best_estimator_
print("Best alpha:", best_alpha)
print("Best MSE:", best_mse)

test_mse = -grid_search.score(X_test, y_test)
print("Test MSE:", test_mse)

kf = KFold(n_splits=10, shuffle=True, random_state=10)

scores = cross_val_score(RR, X_train, y_train, scoring='neg_mean_squared_error', cv=kf, n_jobs=-1)
mean_score = -np.mean(scores)
std_score = np.std(scores)

print("Mean Score:", mean_score)
print("Standard Deviation Score:", std_score)

RR.fit(X_train, y_train)

y_pred = RR.predict(X_test)

MSE = mean_squared_error(y_test, y_pred)
RMSE = mean_squared_error(y_test, y_pred, squared=False)
MAE = mean_absolute_error(y_test, y_pred)
R2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", MSE)
print("Root Mean Squared Error:", RMSE)
print("Mean Absolute Error:", MAE)
print("R2 Score:", R2)