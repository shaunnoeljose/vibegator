# Vibegator - A Mood-Aware Music Recommender 

<p align="center">
  <img src="vibegator_logo.png" alt="VibeGator Logo" width="150" height="150">
</p>

![Streamlit](https://img.shields.io/badge/Framework-Streamlit-ff4b4b?logo=streamlit&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![ML Model](https://img.shields.io/badge/Model-Random%20Forest-blue)
![Made With](https://img.shields.io/badge/Built%20with-Python%203.10-blue?logo=python)

## 🎬 Introduction
In today’s digital streaming era, music platforms are increasingly focused on delivering hyper-personalized listening experiences. While traditional recommender systems often rely on popularity, collaborative filtering, or genre similarity, they fail to account for the emotional and contextual preferences of users. A person studying may prefer quiet, acoustic tracks, while someone working out seeks energetic, high-tempo songs. Integrating mood and listening intent into music recommendations can significantly enhance user satisfaction and engagement.

**VibeGator** is a mood-aware music recommendation engine that blends audio intelligence with emotional context. It uses a **Random Forest classifier** to predict a song’s **mood** (happy/sad) from audio features, and generates recommendations using a **hybrid similarity model** combining textual metadata and acoustic profile similarity.

🎧 Users can fine-tune suggestions using **listening trait filters** like:
- 🎯 *Studying*
- 🧘 *Relaxing*
- 💃 *Partying*
- 🏋️ *Working Out*

All wrapped inside a responsive, Spotify-inspired web interface built with **Streamlit**.

---

## ❓Problem Definition
🎯 Traditional music recommendation systems: Focus heavily on collaborative filtering. Ignore the user’s **emotional state** and **listening context** and offer generic suggestions that miss the mood/intent behind listening

🚀 **VibeGator solves this by:**
- 🧠 Predicting song mood using audio features (valence, tempo, energy, etc.)
- 🎵 Generating hybrid similarity scores (TF-IDF + acoustic cosine)
- 🧩 Filtering results based on listening traits
- 🌐 Delivering real-time, personalized recommendations through an interactive UI

---

## 🔄 VibeGator Workflow

```mermaid
flowchart TD
    A[Input Song and User Filters] --> B[Feature Engineering]
    B --> C[Mood Classification - Random Forest]
    B --> D[Audio and Metadata Processing]
    D --> E[Hybrid Similarity Calculation]
    C --> F[Mood Filtering]
    E --> F
    F --> G[Trait Filtering - Studying or Relaxing]
    G --> H[Final Recommendations]
```

---

## 🖥️ Application mockup

<p align="center">
  <img src="Screenshot 2025-04-18 134119-portrait.png" alt="VibeGator Interface" width="450"/>
</p>

The VibeGator interface allows users to select a song, choose mood preferences, and filter by listening traits — all within an intuitive Streamlit dashboard.

---

## 🌐 Live App

🚀 Try the live demo:  
👉 [**Launch VibeGator on Streamlit Cloud**](https://vibegator-ccagavjaww2b4jw2dzyu5g.streamlit.app/)

> No installation needed — explore song recommendations right in your browser!

---

## 🧪 How to Use the Interface

1. **Select a song** from the dropdown.
2. **Choose a mood preference**:
   - `Auto` will predict the mood of the selected song.
   - Or you can force the system to recommend only `Happy` or `Sad` songs.
3. **Pick a listening trait** *(optional)*:
   - Studying, Relaxing, Party, or Workout.
4. **Select the similarity method**:
   - `Hybrid` (best), `Audio Only`, or `Text Only`.
5. Click **🔍 Get Recommendations** to see your results!

🎯 The model will return a mood-aware, context-driven list of songs that match your selection.

---

✅ *Deployed using [Streamlit Community Cloud](https://streamlit.io/cloud)*  
🔒 All data is processed locally in the session — no user data is stored.


