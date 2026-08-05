# Dysarthria Detection and Speech Therapy App

An AI-based application that detects dysarthria (speech impairment) severity from audio input and provides personalized, interactive speech therapy support using Generative AI.

## Overview

This project combines deep learning-based audio analysis with a GenAI-powered virtual therapist to help individuals with speech impairments track their condition and practice therapy exercises. Related work was presented online at the **Third International Conference on Integration of Advanced Technologies for Industry 5.0 (ICIATI 5.0)**, hosted by KCG College of Technology, Chennai, on 3–4 July 2026.

Developed by a team of 3 students, under the guidance of a faculty mentor.

## Features

- **Severity Classification** – Uses a Convolutional Neural Network (CNN) trained on mel-spectrograms (extracted via Librosa) to classify dysarthria severity from speech audio.
- **AI Virtual Therapist** – Conversational speech therapy support powered by the Groq API (LLaMA 3.3-70B), offering scenario-based practice and real-time feedback.
- **Voice Input** – Accepts spoken input via SpeechRecognition for a natural, hands-free therapy experience.
- **Progress Tracking** – A "Progress Replay" feature lets users compare past and current sessions to track improvement.
- **PDF Report Generation** – Uses pdfplumber to generate and compare progress reports.
- **Phonetic Matching** – A phonetic alternatives dictionary helps match mispronounced words to intended therapy words.

## Tech Stack

| Component | Technology |
|---|---|
| Severity Classification | CNN, Librosa (mel-spectrograms) |
| Conversational AI | Groq API (LLaMA 3.3-70B) |
| Frontend | Streamlit |
| Database | SQLite |
| Report Generation | pdfplumber |
| Voice Input | SpeechRecognition |

## Files in this Repository

- `app.py` – Main Streamlit application (UI and core workflow)
- `main.py` – Core processing/entry logic
- `sound_convert.py` – Audio preprocessing and conversion utilities

> Note: Large assets (trained model weights, audio datasets, generated spectrogram images) are excluded from this repository to keep it lightweight. Available on request.

## Recognition

A related paper, *"A Computational Model For Real Time Speech Severity In Therapy Systems,"* was presented online at **ICIATI 5.0** (Third International Conference on Integration of Advanced Technologies for Industry 5.0), hosted by KCG College of Technology, Chennai, 3–4 July 2026.

## Author

**Jebaprabha B**
B.Tech, Computer Science and Business Systems
KIT – Kalaignarkarunanidhi Institute of Technology, Coimbatore

Co-authors: Tamilarasu P, Sriramsundar K, Vikasni KR
