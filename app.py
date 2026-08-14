import streamlit as st
import pandas as pd
import joblib

# Load files
@st.cache_resource
def load_data():
    df = pd.read_pickle("df.pkl")
    model = joblib.load("spotify_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return df, model, scaler

df, model, scaler = load_data()

st.title("🎵 Spotify Song Recommender")
st.write("Pick a song and get similar recommendations!")

song_name = st.selectbox("Choose a song:", df['name'].values)

if st.button("Recommend"):
    # This part depends on how your model was trained
    # Adjust column names if yours are different
    idx = df[df['name'] == song_name].index[0]
    distances, indices = model.kneighbors(scaler.transform(df.iloc[idx:idx+1].drop(['name','artists'], axis=1)))
    
    st.subheader("You might also like:")
    for i in indices[0][1:6]:
        st.write(f"- {df.iloc[i]['name']} by {df.iloc[i]['artists']}")