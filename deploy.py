import streamlit as st
import pandas as pd
import numpy as np

import os

import ast
from category_encoders import TargetEncoder
import pickle

# Set the CSS styling


# Streamlit app code starts here
st.title("Song Popularity Prediction")
st.write("### Input Data")

col1, col2 = st.columns(2)
Song = col1.text_input("Song Name")
ArtistsGenres = col2.text_input("Artist(s) Genres")
Year = col1.text_input("Album Release Date")
SongLengthms = col2.number_input("Song Length(ms)", min_value=0.0, max_value=2000000.0, step=1.0)
Hot100RankingYear = col1.number_input("Hot100 Ranking Year", min_value=1900, max_value=2024, step=1)
Hot100Rank = col2.number_input("Hot100 Rank", min_value=1, max_value=100, step=1)
Mode = col1.number_input("Mode", min_value=0, max_value=1, step=1)
Acousticness = col2.number_input("Acousticness", min_value=0.0, max_value=1.0, step=0.0001)
Loudness = col1.number_input("Loudness",min_value=-40.0,max_value=0.0, step=0.0001)
Energy = col2.number_input("Energy", min_value=0.0, max_value=1.0, step=0.0001)



data = {
    "Acousticness": [Acousticness],
    "Artist(s) Genres": [ArtistsGenres],
    "Hot100 Ranking Year": [Hot100RankingYear],
    "Mode": [Mode],
    "Album Release Date": [Year],
    "Hot100 Rank": [Hot100Rank],
    "Song Length(ms)": [SongLengthms],
    "Energy": [Energy],
    "Loudness": [Loudness]
}
test_df = pd.DataFrame(data)

test_df["Song"] = [Song]
test_df["Album"] = "Placeholder Album"
test_df["Artist Names"] = "['LeAnn Rimes']"
test_df["Spotify Link"] = "Placeholder Spotify Link"
test_df["Song Image"] = "Placeholder Song Image"
test_df["Spotify URI"] = "Placeholder Spotify URI"
test_df["Popularity"] = -1
test_df["Danceability"] = 0.478
test_df["Instrumentalness"] = 0.000096
test_df["Liveness"] = 0.118
test_df["Speechiness"] = 0.0367
test_df["Tempo"] = 144.705
test_df["Valence"] = 0.564
test_df["Key"] = 7
test_df["Time Signature"] = 4






test_df['Year'] = test_df['Album Release Date'].apply(lambda x: x.split('/')[-1].split('-')[0])


test_df.drop(['Album Release Date'], axis=1, inplace=True)

def convert_to_list(value):
    try:
        return ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return value


test_df['Artist Names'] = test_df['Artist Names'].apply(convert_to_list)
test_df['Artist(s) Genres'] = test_df['Artist(s) Genres'].apply(convert_to_list)
df_exploded = test_df.explode('Artist Names')
df_exploded = df_exploded.explode('Artist(s) Genres')

test_df = df_exploded.copy()

current_directory = os.path.dirname(os.path.abspath(__file__))

pickle_file_path = os.path.join(current_directory, "target_encoder.pkl")

try:
    with open("target_encoder.pkl", 'rb') as f:
        encoder = pickle.load(f)
except Exception as e:
    st.error(f"Error loading target encoder: {e}")


test_data_encoded = encoder.transform(test_df)
current_directory = os.path.dirname(os.path.abspath(__file__))

pickle_file_path = os.path.join(current_directory, "label_encoder.pkl")

try:
    with open(pickle_file_path, 'rb') as f:
        le = pickle.load(f)
except Exception as e:
    st.error(f"Error loading target encoder: {e}")




train_unique_labels = set(le.classes_)
test_data_encoded['Artist Names'] = test_data_encoded['Artist Names'].apply(
    lambda x: le.transform([x])[0] if x in train_unique_labels else -1)

test_df = test_data_encoded.copy()


def aggregate_rows(group):
    sum_artists = sum(group['Artist Names'].unique())
    sum_genres = sum(group['Artist(s) Genres'].unique())

    return pd.Series({
        'Artist Names Encoded': sum_artists,
        'Artist(s) Genres Encoded': sum_genres
    })


aggregated_df = test_df.groupby(test_df.index).apply(aggregate_rows)

aggregated_df = aggregated_df.reset_index(drop=True)
original_indices = set(test_df.index)
aggregated_indices = set(aggregated_df.index)

missing_indices = original_indices - aggregated_indices
extra_indices = aggregated_indices - original_indices
test_df = pd.concat([aggregated_df, test_df], axis=1)
columns_to_drop = ['Artist Names', 'Artist(s) Genres', 'Song', 'Album', 'Spotify Link', 'Song Image', 'Spotify URI']

test_df.drop(columns_to_drop, axis=1, inplace=True)
test_df.reset_index(drop=True, inplace=True)
column_names = {
    'Artist Names Encoded': 'Artist Names',
    'Artist(s) Genres Encoded': 'Artist(s) Genres'
}

# Assuming df is your DataFrame
first_row_df = test_df.iloc[[0]].copy()  # Selecting only the first row and making a copy
test_df = first_row_df  # Replacing the original DataFrame with the DataFrame containing only the first row


test_df = test_df.rename(columns=column_names)
test_df['Hype']=test_df['Loudness']+test_df['Energy']
test_df['Happiness'] = test_df['Danceability'] + test_df['Valence']
offset = 1e-10

test_df['Liveness_log'] = np.log(test_df['Liveness'] + offset)

# In[170]:


test_df['Acousticness_sqrt'] = np.sqrt(test_df['Acousticness'] + offset)

# In[171]:


test_df['Speechiness_sqrt'] = np.sqrt(test_df['Speechiness'])

# In[172]:


columns_to_drop = ['Liveness', 'Acousticness', 'Speechiness']

test_df.drop(columns_to_drop, axis=1, inplace=True)
test_df.reset_index(drop=True, inplace=True)
column_names = {
    'Liveness_log': 'Liveness',
    'Acousticness_sqrt': 'Acousticness',
    'Speechiness_sqrt': 'Speechiness'
}

test_df = test_df.rename(columns=column_names)
test_features = test_df.drop(['Popularity'],axis=1)
test_features_df = test_features[['Artist Names', 'Artist(s) Genres', 'Year', 'Hot100 Ranking Year',
                                  'Hot100 Rank', 'Song Length(ms)', 'Danceability', 'Energy',
                                  'Instrumentalness', 'Loudness', 'Tempo', 'Valence', 'Key', 'Mode',
                                  'Time Signature', 'Hype','Happiness', 'Liveness',
                                  'Acousticness', 'Speechiness']]



current_directory = os.path.dirname(os.path.abspath(__file__))

pickle_file_path = os.path.join(current_directory, "scaler.pkl")

try:
    with open(pickle_file_path, 'rb') as f:
        scaler = pickle.load(f)
except Exception as e:
    st.error(f"Error loading target encoder: {e}")

test_features_scaled = scaler.transform(test_features_df)
test_features_scaled_df = pd.DataFrame(test_features_scaled, columns=test_features_df.columns)
selected_features = ['Acousticness', 'Artist(s) Genres', 'Hot100 Ranking Year', 'Mode', 'Year', 'Hot100 Rank',
                     'Song Length(ms)', 'Hype']

test_features_scaled_df_selected = test_features_scaled_df[selected_features]
test_features = test_features_scaled_df_selected.copy()



current_directory = os.path.dirname(os.path.abspath(__file__))

pickle_file_path = os.path.join(current_directory, "xgb_regressor_model.pkl")

try:
    with open(pickle_file_path, 'rb') as f:
        loaded_xgb_regressor = pickle.load(f)
except Exception as e:
    st.error(f"Error loading target encoder: {e}")


# Assuming y_pred is the scaled prediction with shape (4, 20)
def load_and_predict():
    # Load the model
    y_pred = loaded_xgb_regressor.predict(test_features)

    current_directory = os.path.dirname(os.path.abspath(__file__))

    pickle_file_path = os.path.join(current_directory, "y_scaler.pkl")

    try:
        with open(pickle_file_path, 'rb') as f:
            y_scaler = pickle.load(f)
    except Exception as e:
        st.error(f"Error loading target encoder: {e}")

    y_pred_original_scale = y_scaler.inverse_transform([y_pred]).round(0)
    st.write("### Song Popularity out of 100:")
    st.write(f'# {y_pred_original_scale[0, 0]}')


# Button to trigger the prediction
if st.button('Run Prediction'):
    load_and_predict()



import base64

def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Define function to set background image
def set_background(png_file):
    bin_str = get_base64_of_bin_file(png_file)
    page_bg_img = f'''
    <style>
    [data-testid="stAppViewContainer"] {{
        background-image: url("data:image/png;base64,{bin_str}");
        background-size: cover;
        color: white; /* Set text color to white */
    }}
    
    </style>
    '''
    st.markdown(page_bg_img, unsafe_allow_html=True)

# Set background image with the correct file path
set_background(r"dd.png")

