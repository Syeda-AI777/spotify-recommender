import streamlit as st
import pandas as pd
import pickle
import gdown
import os

DF_FILE = "df.pkl"
DF_URL = "https://drive.google.com/uc?id=17lzURcBBOUlN47hny_E1hwZxyo6DLxP3"

# Force delete old corrupted file and re-download
if os.path.exists(DF_FILE):
    os.remove(DF_FILE)

st.info("Downloading data file... please wait 10 seconds")
gdown.download(DF_URL, DF_FILE, quiet=False)

# Load files
with open(DF_FILE, 'rb') as f:
    df = pickle.load(f)
    
scaler = pickle.load(open('scaler.pkl', 'rb'))
model = pickle.load(open('spotify_model.pkl', 'rb'))

st.title("Spotify Music Recommender 🎵")
st.success("App loaded successfully!")
