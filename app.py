import streamlit as st
import pandas as pd
import numpy as np
import pickle
import requests
import io
from sklearn.neighbors import NearestNeighbors

st.set_page_config(page_title="Spotify Song Recommender", layout="wide")

# Google Drive File IDs
FILE_IDS = {
    "df": "1CaGTPvC3oxYeEwtUgy9W4kQyMnNCCvQo",
    "scaler": "1_8zPdNlLHGvpsZFp8AEiTDJsOf4emgPJ",
    "model": "1KO1d2VbhqLd1tQoAqFsrMpqoZHL7SKyK"
}

@st.cache_resource
def download_file(file_id):
    URL = "https://drive.google.com/uc?export=download&id="
    session = requests.Session()
    response = session.get(URL + file_id)
    return io.BytesIO(response.content)

@st.cache_resource
def load_data():
    with st.spinner('Downloading models from Drive... 30 seconds'):
        df_bytes = download_file(FILE_IDS["df"])
        df = pd.read_pickle(df_bytes)

        scaler_bytes = download_file(FILE_IDS["scaler"])
        scaler = pickle.load(scaler_bytes)

        model_bytes = download_file(FILE_IDS["model"])
        model = pickle.load(model_bytes)
    return df, scaler, model

df, scaler, model = load_data()
st.success('Models Loaded Successfully!')

features = ['danceability', 'energy', 'key', 'loudness', 'mode',
            'speechiness', 'acousticness', 'instrumentalness',
            'liveness', 'valence', 'tempo']

st.title("🎵 Spotify Song Recommendation System")
st.write("Select a song and get 10 similar songs based on audio features")

song_list = df['track_name'].values
selected_song = st.selectbox("Type or select a song:", song_list)

if st.button('Get Recommendations'):
    with st.spinner('Finding similar songs...'):
        song_data = df[df['track_name'] == selected_song][features]
        song_scaled = scaler.transform(song_data)

        distances, indices = model.kneighbors(song_scaled)

        st.subheader(f"Top 10 songs similar to '{selected_song}':")
        for i in indices[0][1:11]:
            st.write(f"**{df['track_name'].iloc[i]}** - *{df['artists'].iloc[i]}*")
