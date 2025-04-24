import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity

# streamlit setup
st.set_page_config(page_title="🎵VibeGator Recommender", layout="centered")

st.markdown("""
    <style>
    html, body, .stApp {
        background-color: #191414 !important;
        color: white !important;
    }
    h1, h2, h3, h4, h5, h6,
    .stMarkdown h1, .stMarkdown h2 {
        color: white !important;
    }
    label, .css-1cpxqw2, .css-16idsys, .st-emotion-cache, .st-emotion-cache-1rs6os {
        color: white !important;
        font-weight: 500 !important;
    }
    .stRadio div, .stRadio label, .stRadio div > label {
        color: white !important;
    }
    .stImageCaption, .css-1n76uvr {
        color: white !important;
        font-weight: 600 !important;
        text-align: center;
    }
    .stButton > button {
        background-color: #1DB954 !important;
        color: white !important;
        border: none;
    }
    .stDataFrame {
        background-color: #222 !important;
        color: white !important;
    }
    </style>
""", unsafe_allow_html=True)

st.image("Assets/vibegator_logo.png", width=120)
st.title("🎧 VibeGator")
st.markdown(
    "<h5 style='color: white; font-size: 20px;'>Mapping User Sentiment Into Curation</h5>",
    unsafe_allow_html=True
)
 
# loading the dataset
@st.cache_data
def load_data():
    df = pd.read_csv("Data/final_music.csv")
    df.dropna(subset=['trackName', 'artistName', 'genre',
                      'danceability', 'energy', 'loudness', 'speechiness',
                      'acousticness', 'instrumentalness', 'liveness',
                      'valence', 'tempo'], inplace=True)
    return df

df = load_data()

# feature engineering
def bucket_feature(val, bins, labels):
    return pd.cut(val, bins=bins, labels=labels, include_lowest=True).astype(str)

df['valence_bucket'] = bucket_feature(df['valence'], [0, 0.33, 0.66, 1], ['low_valence', 'medium_valence', 'high_valence'])
df['energy_bucket'] = bucket_feature(df['energy'], [0, 0.33, 0.66, 1], ['low_energy', 'medium_energy', 'high_energy'])
df['tempo_bucket'] = bucket_feature(df['tempo'], [0, 100, 140, 250], ['slow', 'medium', 'fast'])

def bucket_loudness(val):
    return 'quiet' if val < -20 else 'moderate' if val < -10 else 'loud'

def bucket_acousticness(val):
    return 'acoustic' if val > 0.7 else 'balanced' if val > 0.3 else 'synthetic'

df['loudness_bucket'] = df['loudness'].apply(bucket_loudness)
df['acousticness_bucket'] = df['acousticness'].apply(bucket_acousticness)

def assign_listening_traits(row):
    traits = []
    if row['loudness_bucket'] == 'quiet' and row['acousticness_bucket'] == 'acoustic':
        traits.extend(['studying', 'relaxing'])
    elif row['loudness_bucket'] == 'loud' and row['acousticness_bucket'] == 'synthetic':
        traits.extend(['workout', 'party'])
    return ' '.join(traits)

df['listening_traits'] = df.apply(assign_listening_traits, axis=1)

df['combined_features'] = (
    df['genre'].fillna('') + ' ' +
    df['artistName'].fillna('') + ' ' +
    df['trackName'].fillna('') + ' ' +
    df['valence_bucket'] + ' ' +
    df['energy_bucket'] + ' ' +
    df['tempo_bucket']
)

# calculating the cosine similarity
@st.cache_resource
def compute_similarities():
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['combined_features'])
    scaler = MinMaxScaler()
    audio_scaled = scaler.fit_transform(df[['danceability', 'energy', 'loudness', 'speechiness',
                                            'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']])
    text_sim = cosine_similarity(tfidf_matrix)
    audio_sim = cosine_similarity(audio_scaled)
    hybrid_sim = (text_sim + audio_sim) / 2
    return text_sim, audio_sim, hybrid_sim

text_sim, audio_sim, hybrid_sim = compute_similarities()

# loading the models and the pipeline
@st.cache_resource
def load_models():
    pipeline = joblib.load("Code/classification_pipeline.pkl")
    model = joblib.load("Code/rf_model.pkl")
    return pipeline, model

pipeline, rf_model = load_models()

def predict_mood_rf(df_subset):
    processed = pipeline.transform(df_subset.copy())
    preds = rf_model.predict(processed)
    return ['happy' if p == 1 else 'sad' for p in preds]

# recommendation function
def get_recommendations(song_title, sim_type="Hybrid", mood_pref="Auto", trait_choice="", top_n=10):
    sim_matrix = {"Hybrid": hybrid_sim, "Text Only": text_sim, "Audio Only": audio_sim}.get(sim_type, hybrid_sim)

    idx_series = df[df['trackName'].str.lower() == song_title.lower()].index
    if idx_series.empty:
        return pd.DataFrame(), "Song not found"
    idx = idx_series[0]

    if idx >= len(sim_matrix):
        return pd.DataFrame(), "Index mismatch. Try another song."

    sim_scores = sorted(list(enumerate(sim_matrix[idx])), key=lambda x: x[1], reverse=True)
    rec_indices = [i[0] for i in sim_scores[1:top_n+1] if i[0] < len(df)]

    recommended = df.iloc[rec_indices].copy()
    recommended['predicted_mood'] = predict_mood_rf(recommended)

    if mood_pref == "Auto":
        base_mood = predict_mood_rf(df.iloc[[idx]])[0]
        filtered = recommended[recommended['predicted_mood'] == base_mood]
    else:
        filtered = recommended[recommended['predicted_mood'] == mood_pref.lower()]

    if trait_choice:
        filtered = filtered[filtered['listening_traits'].str.contains(trait_choice)]
        if filtered.empty:
            return recommended[['trackName', 'artistName', 'genre', 'predicted_mood']], " No results match both mood and trait. Showing mood-only results."

    return filtered[['trackName', 'artistName', 'genre', 'predicted_mood']], None

# filtering the valid songs
def get_valid_songs_for_trait(trait, limit=100):
    valid_songs = []
    subset = df[df['listening_traits'].str.contains(trait)] if trait else df
    for song in subset['trackName'].unique():
        idx_series = df[df['trackName'].str.lower() == song.lower()].index
        if idx_series.empty:
            continue
        idx = idx_series[0]
        if idx >= len(hybrid_sim):
            continue
        sim_scores = sorted(list(enumerate(hybrid_sim[idx])), key=lambda x: x[1], reverse=True)
        rec_indices = [i[0] for i in sim_scores[1:11] if i[0] < len(df)]
        if not rec_indices:
            continue
        rec_df = df.iloc[rec_indices]
        if trait and rec_df['listening_traits'].str.contains(trait).any():
            valid_songs.append(song)
        elif not trait:
            valid_songs.append(song)
        if len(valid_songs) >= limit:
            break
    return sorted(valid_songs)

# caching for All (No Filter)
@st.cache_data
def get_all_valid_songs(limit=100):
    return get_valid_songs_for_trait("", limit=limit)

trait_choice = st.selectbox("🎯 Listening Trait", ["All (No Filter)", "studying", "relaxing", "party", "workout"])
sim_method = st.radio("🔁 Similarity", ["Hybrid", "Text Only", "Audio Only"], horizontal=True)
mood_choice = st.radio("🎚️ Mood", ["Auto", "Happy", "Sad"], horizontal=True)

# including a slider
max_limit = len(df)
default_limit = min(100, max_limit) 
limit_value = st.slider(
    "🔢 Max Songs to Search for Recommendations",
    min_value=10,
    max_value=max_limit,
    value=default_limit,
    step=10
)

# controlling the song dropdown list
if trait_choice == "All (No Filter)":
    song_options = get_all_valid_songs(limit=limit_value)
    trait_choice = ""
else:
    song_options = get_valid_songs_for_trait(trait_choice, limit=limit_value)

if song_options:
    song_selected = st.selectbox("🎵 Select a Song", song_options)
else:
    song_selected = None
    st.warning(" No valid songs found for this trait.")

# calling recommendation
if song_selected and st.button("🔍 Get Recommendations"):
    recs, error = get_recommendations(song_selected, sim_method, mood_choice, trait_choice)
    if error:
        st.warning(error)
    elif recs.empty:
        st.info("No mood-matching recommendations found.")
    else:
        st.dataframe(recs.reset_index(drop=True))
