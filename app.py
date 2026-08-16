import streamlit as st
import pandas as pd
import pickle
import requests
import os

DF_FILE = "df.pkl"
FILE_ID = "17lzURcBBOUlN47hny_E1hwZxyo6DLxP3"

# Download with confirmation token to bypass Google virus scan
def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()
    response = session.get(URL, params={'id': id}, stream=True)
    
    # Check for confirmation token
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            params = {'id': id, 'confirm': value}
            response = session.get(URL, params=params, stream=True)
    
    with open(destination, "wb") as f:
        for chunk in response.iter_content(32768):
            if chunk:
                f.write(chunk)

st.info("Downloading data file... please wait 20 seconds")
download_file_from_google_drive(FILE_ID, DF_FILE)

# Load files
with open(DF_FILE, 'rb') as f:
    df = pickle.load(f)
    
scaler = pickle.load(open('scaler.pkl', 'rb'))
model = pickle.load(open('spotify_model.pkl', 'rb'))

st.title("Spotify Music Recommender 🎵")
st.dataframe(df.head())
st.success("App loaded successfully!")
