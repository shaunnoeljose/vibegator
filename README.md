# Spotify A Mood-Aware Music Recommender - Vibegator
![banner](vibegator_logo.png)
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

## ❓Problem Definition
🎯 Traditional music recommendation systems: Focus heavily on collaborative filtering. Ignore the user’s **emotional state** and **listening context** and offer generic suggestions that miss the mood/intent behind listening

🚀 **VibeGator solves this by:**
- 🧠 Predicting song mood using audio features (valence, tempo, energy, etc.)
- 🎵 Generating hybrid similarity scores (TF-IDF + acoustic cosine)
- 🧩 Filtering results based on listening traits
- 🌐 Delivering real-time, personalized recommendations through an interactive UI





