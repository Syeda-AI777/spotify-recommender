# Spotify Song Recommendation System

## Description
A ML-powered web app that recommends 10 similar songs based on audio features 
like danceability, energy, valence using K-Nearest Neighbors.

## Tech Stack
Python, Streamlit, Scikit-learn, Pandas, Google Drive

## How it works
1. Dataset of 50k songs with audio features
2. StandardScaler + KNN model trained on features
3. User selects a song → model finds 10 closest songs by cosine similarity

## How to run
1. Clone repo
2. pip install streamlit pandas scikit-learn requests
3. streamlit run app.py

## Live Demo
https://bc56gwy3pzjnrgd3rxoy4s.streamlit.app
