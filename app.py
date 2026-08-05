# app.py
import os
import re
import gc
import random
import hashlib
import sqlite3
import time
from datetime import datetime
from io import BytesIO
import pyttsx3
import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
import soundfile as sf
import librosa
import streamlit as st
from tensorflow.keras.models import load_model
import cv2
import pandas as pd
import streamlit as st
from gtts import gTTS
import tempfile
import os
# Speech: recognition + TTS
import speech_recognition as sr
from gtts import gTTS
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Image

# -----------------------------
# SESSION-STATE INITIALIZATION
# -----------------------------
if "user" not in st.session_state:
    st.session_state.user = None
if "therapy_active" not in st.session_state:
    st.session_state.therapy_active = False
if "therapy_word" not in st.session_state:
    st.session_state.therapy_word = ""
if "attempts_left" not in st.session_state:
    st.session_state.attempts_left = 0




if "last_speech_result" not in st.session_state:
    st.session_state.last_speech_result = None
if "new_speech_entry" not in st.session_state:
    st.session_state.new_speech_entry = False
# -----------------------------
# CONFIG (audio / model)
# -----------------------------
class Conf:
    sampling_rate = 44100
    duration = 1
    hop_length = 350 * duration
    fmin = 1
    fmax = sampling_rate // 2
    n_mels = 256
    n_fft = n_mels * 20
    samples = sampling_rate * duration

IMG_SIZE = 50
DATASET_DIR = "dataset"
CATEGORIES = sorted(os.listdir(DATASET_DIR)) if os.path.isdir(DATASET_DIR) else []
MODEL_PATH = "CNN.model"

os.makedirs("uploads", exist_ok=True)
os.makedirs("spectrograms", exist_ok=True)
os.makedirs("tts_cache", exist_ok=True)

st.set_page_config(page_title="AI Speech Therapy", page_icon="🗣️", layout="wide")

# -----------------------------
# DATABASE (SQLite)
# -----------------------------
conn = sqlite3.connect("therapy_app.db", check_same_thread=False)
c = conn.cursor()

def init_db():
    c.execute("""
    CREATE TABLE IF NOT EXISTS users(
    username TEXT PRIMARY KEY,
    password TEXT NOT NULL,
    points INTEGER DEFAULT 0
    )
    """)
    c.execute("""
    CREATE TABLE IF NOT EXISTS results(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT NOT NULL,
        ts TEXT NOT NULL,
        word TEXT,
        confidence REAL,
        severity TEXT,
        source TEXT,
        pred_class TEXT,
        spectrogram_path TEXT,
        FOREIGN KEY(username) REFERENCES users(username)
    )
    """)
    conn.commit()

def hash_pw(p: str) -> str:
    return hashlib.sha256(p.encode()).hexdigest()

def add_user(username, password_hash):
    try:
        c.execute("INSERT INTO users(username,password) VALUES (?,?)",
                  (username, password_hash))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False

def get_user(username):
    c.execute("SELECT username,password FROM users WHERE username=?",(username,))
    return c.fetchone()

def add_result(username, word, confidence, severity, source, pred_class, spectrogram_path):
    c.execute("""
        INSERT INTO results(username,ts,word,confidence,severity,source,pred_class,spectrogram_path)
        VALUES (?,?,?,?,?,?,?,?)
    """, (username, datetime.now().isoformat(timespec='seconds'),
          word, float(confidence), severity, source, pred_class, spectrogram_path))
    conn.commit()

def fetch_results_for_user(username, last_n=None):
    conn.commit()  # ✅ force latest DB read
    if last_n:
        c.execute("""
            SELECT ts, pred_class, confidence, severity, source, spectrogram_path
            FROM results WHERE username=? AND source IN ('Upload','Record')
            ORDER BY ts DESC LIMIT ?
        """, (username, last_n))
    else:
        c.execute("""
            SELECT ts, pred_class, confidence, severity, source, spectrogram_path
            FROM results WHERE username=? AND source IN ('Upload','Record')
            ORDER BY ts DESC
        """, (username,))
    return c.fetchall()

def get_points(username):
    c.execute("SELECT points FROM users WHERE username=?", (username,))
    result = c.fetchone()
    return result[0] if result else 0


def update_points(username, pts):
    c.execute("UPDATE users SET points = points + ? WHERE username=?", (pts, username))
    conn.commit()

init_db()

try:
    c.execute("ALTER TABLE users ADD COLUMN points INTEGER DEFAULT 0")
    conn.commit()
except:
    pass
# -----------------------------
# Load CNN model
# -----------------------------
@st.cache_resource(show_spinner=False)
def load_cnn_model(path):
    if os.path.exists(path):
        return load_model(path)
    return None

model = load_cnn_model(MODEL_PATH)

def confidence_to_severity(conf):
    if conf < 0.40: return "Mild"
    elif conf < 0.70: return "Moderate"
    else: return "Severe"

# -----------------------------
# Audio & Spectrogram helpers
# -----------------------------
def read_audio_from_any(path_or_file, trim_long_data=True):
    try:
        y, sr = librosa.load(path_or_file, sr=Conf.sampling_rate, mono=True)
    except Exception as e:
        st.error(f"Could not load audio: {e}")
        return None
    if y is None or len(y) == 0: return None
    y, _ = librosa.effects.trim(y)
    if len(y) > Conf.samples:
        if trim_long_data: y = y[:Conf.samples]
    else:
        pad = Conf.samples - len(y)
        off = pad // 2
        y = np.pad(y, (off, Conf.samples - len(y) - off), 'constant')
    return y

def audio_to_mel(y):
    S = librosa.feature.melspectrogram(y=y, sr=Conf.sampling_rate, n_mels=Conf.n_mels,
                                       hop_length=Conf.hop_length, n_fft=Conf.n_fft,
                                       fmin=Conf.fmin, fmax=Conf.fmax)
    return librosa.power_to_db(S).astype(np.float32)

def save_mel_as_image(mel, out_path):
    plt.figure(figsize=(3, 3), dpi=100)
    plt.imshow(mel, cmap="gray", aspect="auto", origin="lower")

    #plt.imshow(, aspect="auto", origin="lower")
    plt.axis("off")
    plt.savefig(out_path, bbox_inches='tight', pad_inches=0)
    plt.close()

def audio_to_model_ready_image_file(y, base_name):
    mel = audio_to_mel(y)
    spec_path = os.path.join("spectrograms", f"{base_name}.jpg")
    save_mel_as_image(mel, spec_path)
    img = cv2.imread(spec_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    return img, spec_path

def predict_from_img_array(img_array):
    if model is None:
        return "Unknown", 0.0
    X = np.array(img_array, dtype=np.float32).reshape(-1, IMG_SIZE, IMG_SIZE, 1) / 255.0
    pred = model.predict(X, verbose=0)[0]
    idx = int(np.argmax(pred))
    return CATEGORIES[idx] if CATEGORIES else str(idx), float(np.max(pred))

# -----------------------------
# Recording helper
# -----------------------------
def record_audio_wav(duration_s=2, fs=Conf.sampling_rate, filename="recorded.wav"):
    st.info(f"Recording {duration_s} seconds…")
    recording = sd.rec(int(duration_s*fs), samplerate=fs, channels=1, dtype='float32')
    sd.wait()
    sf.write(filename, recording, fs)
    return filename

# -----------------------------
# TTS (gTTS) for browser playback
# -----------------------------
import streamlit as st
import pyttsx3
import tempfile
import os

# --- Text to speech function ---
# --- Text to speech function ---
def speak_text(text):
    try:
        # Convert text to speech with gTTS
        tts = gTTS(text=text, lang="en")
        
        # Save audio to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
            tts.save(tmp.name)
            tmp_path = tmp.name

        # Play the audio in Streamlit
        audio_file = open(tmp_path, "rb")
        audio_bytes = audio_file.read()
        st.audio(audio_bytes, format="audio/mp3")

        # Cleanup
        audio_file.close()
        os.remove(tmp_path)

    except Exception as e:
        st.warning(f"⚠️ Audio error: {e}")
        st.info(f"(Please say: {text})")

# -----------------------------
# Speech recognition (microphone)
# -----------------------------
def capture_speech(duration=3):
    """Listen from microphone and return recognized text (or empty string)."""
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        # brief notice to user
        st.info("Listening... please speak now")
        try:
            audio = recognizer.listen(source, timeout=duration, phrase_time_limit=duration)
        except Exception:
            return ""
    try:
        return recognizer.recognize_google(audio)
    except Exception:
        return ""

def normalize_phrase(s):
    if not s: return ""
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    return re.sub(r"\s+"," ",s).strip()

# -----------------------------
# UI / Auth
# -----------------------------
st.title("🗣️ AI-Based Speech Therapy")

with st.sidebar:
    st.header("Account")
    auth_mode = st.radio("Authentication", ["Login", "Sign up"])
    if auth_mode == "Sign up":
        su_user = st.text_input("New username")
        su_pw = st.text_input("New password", type="password")
        if st.button("Create account"):
            if not su_user or not su_pw:
                st.error("Required fields missing.")
            elif get_user(su_user):
                st.error("User exists.")
            else:
                if add_user(su_user, hash_pw(su_pw)):
                    st.success("Account created. Please login.")
    st.markdown("---")
    li_user = st.text_input("Username")
    li_pw = st.text_input("Password", type="password")
    login_clicked = st.button("Login")

if login_clicked:
    row = get_user(li_user)
    if row and row[1] == hash_pw(li_pw):
        st.session_state.user = row[0]
        st.success(f"Logged in as {st.session_state.user}")
    else:
        st.error("Invalid login")

if not st.session_state.user:
    st.info("Please log in (or sign up) from the left panel to continue.")
    st.stop()

# -----------------------------
# Tabs

#THERAPY_WORDS = ["Good morning", "Good night", "How are you", "Hello", "Thank you"]
# 🎯 Personalized therapy sets (UPDATED)
SEVERE_SET = [
    "A", "B", "C", "D", "E",
    "Ma", "Pa", "Ba", "La", "Ta"
]

MODERATE_SET = [
    "Hello there",
    "Drink water",
    "I am happy",
    "Good job",
    "Come here"
]

MILD_SET = [
    "Good morning, how are you",
    "I am feeling very happy today",
    "Nice to meet you",
    "Thank you very much",
    "Have a great day"
]


def start_new_round():
    severity = st.session_state.get("last_severity")

    if severity == "Severe":
        word_list = SEVERE_SET
    elif severity == "Moderate":
        word_list = MODERATE_SET
    elif severity == "Mild":
        word_list = MILD_SET
    else:
        return  # ❌ no therapy if normal

    st.session_state.therapy_word = random.choice(word_list)
    st.session_state.attempts_left = 3
    st.session_state.attempt = 1
    st.session_state.therapy_active = True
    st.session_state.last_result = None

if "therapy_active" not in st.session_state:
    st.session_state.therapy_active = False
# -----------------------------
tabs = st.tabs(["Dashboard","Analytics","Speech Assessment","Therapy Exercises","Download Report","Compare Reports","Virtual Therapist"])


def generate_pdf(df, username):
    file_path = f"{username}_report.pdf"

    doc = SimpleDocTemplate(file_path)
    styles = getSampleStyleSheet()
    elements = []

    import matplotlib.pyplot as plt
    from reportlab.platypus import Image

    # =========================
    # 📈 Confidence Graph
    # =========================
    graph_path = f"{username}_graph.png"

    plt.figure()
    plt.plot(df["Confidence"].values, marker='o')
    plt.title("Confidence Trend")
    plt.xlabel("Test Number")
    plt.ylabel("Confidence")
    plt.savefig(graph_path)
    plt.close()

    # =========================
    # 📊 Severity Pie Chart
    # =========================
    severity_path = f"{username}_severity.png"

    severity_counts = df["Severity"].value_counts()

    plt.figure()
    plt.pie(severity_counts, labels=severity_counts.index, autopct='%1.1f%%')
    plt.title("Severity Distribution")
    plt.savefig(severity_path)
    plt.close()

    # =========================
    # 🧠 Title
    # =========================
    elements.append(Paragraph("AI Speech Therapy Report", styles['Title']))
    elements.append(Spacer(1, 10))

    elements.append(Paragraph(f"User: {username}", styles['Normal']))
    elements.append(Spacer(1, 10))

    # =========================
    # 📊 Summary
    # =========================
    avg_conf = df["Confidence"].mean()
    total_tests = len(df)
    most_common_severity = df["Severity"].mode()[0]

    elements.append(Paragraph(f"Total Tests: {total_tests}", styles['Normal']))
    elements.append(Paragraph(f"Average Confidence: {avg_conf:.2f}", styles['Normal']))
    elements.append(Paragraph(f"Most Frequent Severity: {most_common_severity}", styles['Normal']))
    elements.append(Spacer(1, 15))

    

    # =========================
    # 📈 Confidence Graph
    # =========================
    elements.append(Paragraph("Confidence Trend Graph", styles['Heading2']))
    elements.append(Spacer(1, 10))
    elements.append(Image(graph_path, width=400, height=200))
    elements.append(Spacer(1, 20))

    # =========================
    # 📊 Severity Pie Chart
    # =========================
    elements.append(Paragraph("Severity Distribution", styles['Heading2']))
    elements.append(Spacer(1, 10))
    elements.append(Image(severity_path, width=300, height=200))
    elements.append(Spacer(1, 20))

    # =========================
    # 📋 Table
    # =========================
    elements.append(Paragraph("Table", styles['Heading2']))

    table_data = [list(df.columns)] + df.values.tolist()

    table = Table(table_data)
    table.setStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.grey),
        ('TEXTCOLOR',(0,0),(-1,0),colors.whitesmoke),
        ('GRID',(0,0),(-1,-1),1,colors.black)
    ])

    elements.append(table)
    elements.append(Spacer(1, 20))
    
    # =========================
    # 📊 Improvement Insight (NEW)
    # =========================
    if len(df) >= 2:
        first = df["Confidence"].iloc[-1]
        last = df["Confidence"].iloc[0]

        if last > first:
            trend = "Improvement detected over time."
        elif last < first:
            trend = "Slight decline observed. More practice recommended."
        else:
            trend = "No significant change observed."

        elements.append(Paragraph("Progress Insight:", styles['Heading2']))
        elements.append(Paragraph(trend, styles['Normal']))
        elements.append(Spacer(1, 15))

    # =========================
    # 🧾 Final Insight
    # =========================
    if avg_conf > 0.7:
        remark = "Speech condition appears severe. Regular therapy is highly recommended."
    elif avg_conf > 0.4:
        remark = "Moderate speech difficulty detected. Consistent practice can improve speech clarity."
    else:
        remark = "Mild or no significant speech issues detected."

    elements.append(Paragraph("Final Assessment:", styles['Heading2']))
    elements.append(Paragraph(remark, styles['Normal']))

    doc.build(elements)

    return file_path

# ---------------- Dashboard ----------------
with tabs[0]:
    st.subheader("📊 Progress Dashboard (Last 5 CNN Results)")
    data = fetch_results_for_user(st.session_state.user, last_n=5)
    if not data:
        st.info("No history yet.")
    else:
        rows = []
        for d in data:
            # d = (ts, pred_class, confidence, severity, source, spec_path)
            rows.append({
                "Time": d[0],
                "Predicted": d[1],
                "Confidence": round(d[2],3),
                "Severity": d[3],
                "Source": d[4],
                "Spectrogram": d[5] or ""
            })
        df = pd.DataFrame(rows)
        st.dataframe(df[["Time","Predicted","Confidence","Severity","Source"]])

        # Show thumbnails below
        for r in rows:
            if r["Spectrogram"]:
                st.image(r["Spectrogram"], caption=f"{r['Time']} — {r['Predicted']}", width=300)

    # rerun only when a new entry is detected
    if "new_entry" in st.session_state and st.session_state.new_entry:
        st.session_state.new_entry = False
        st.rerun()

#------------------Analytics------------------
# ---------------- Analytics ----------------
with tabs[1]:
    st.markdown("## 📊 Detailed Analytics")
    

    data = fetch_results_for_user(st.session_state.user, last_n=30)
    
    if not data:
        st.info("No data available yet.")
    
    else:
        #st.write("Total records:", len(data))
        rows = []
        for d in data:
            rows.append({
                "Time": d[0],
                "Predicted": d[1],
                "Confidence": round(d[2], 3),
                "Severity": d[3],
                "Source": d[4]
            })

        df = pd.DataFrame(rows)

        df["Time"] = pd.to_datetime(df["Time"])
        if st.button("🔄 Refresh Analytics"):
            st.rerun()

        # =========================
        # 📈 Confidence Graph
        # =========================
        st.subheader("📈 Confidence Over Time")

        import matplotlib.pyplot as plt
        plt.figure(figsize=(12,4))

        df = df.sort_values("Time")  # ✅ important

        plt.plot(df["Time"], df["Confidence"], marker='o')

        plt.xticks(rotation=45)
        plt.gcf().autofmt_xdate()  # ✅ fixes date display

        plt.xlabel("Date")
        plt.ylabel("Confidence")
        plt.title("Confidence Trend")

        plt.tight_layout()

        st.pyplot(plt)
        # =========================
        # 📊 Severity Graph
        # =========================
        st.subheader("📊 Severity Distribution")

        plt.figure(figsize=(8,4))

        severity_counts = df["Severity"].value_counts()

        severity_order = ["Mild", "Moderate", "Severe", "Normal"]
        severity_counts = severity_counts.reindex(severity_order).fillna(0)

        plt.bar(severity_counts.index, severity_counts.values)

        plt.xlabel("Severity")
        plt.ylabel("Count")
        plt.title("Severity Distribution")

        # ✅ FIX: only integer values on Y-axis
        plt.yticks(range(0, int(max(severity_counts.values)) + 1))

        plt.grid(axis='y')

        plt.tight_layout()

        st.pyplot(plt)

        # =========================
        # 📊 Raw Data Table
        # =========================
        st.subheader("📋 Full Data")
        st.dataframe(df)

# ---------------- Assessment ----------------
with tabs[2]:
    st.subheader("🎤 Speech Assessment (CNN Model)")
    mode = st.radio("Input method", ["Upload", "Record"])

    def final_decision(pred, conf):
        conf = round(conf, 2)
        pred_lower = pred.lower()

        # ✅ First trust model output
        if "non" in pred_lower or "normal" in pred_lower:
            return "Normal", None

        # ✅ If model says dysarthria, classify severity
        if conf > 0.90:
            return "Dysarthria", "Severe"
        elif conf > 0.80:
            return "Dysarthria", "Moderate"
        else:
            return "Dysarthria", "Mild"

    # -------- Upload --------
    if mode == "Upload":
        file = st.file_uploader("Upload file", type=["wav", "mp3", "flac", "m4a"])
        if file:
            y = read_audio_from_any(file)
            if y is not None:
                base = f"{st.session_state.user}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                wav_path = os.path.join("uploads", f"{base}.wav")
                sf.write(wav_path, y, Conf.sampling_rate)

                img, spec = audio_to_model_ready_image_file(y, base)
                pred, conf = predict_from_img_array(img)

                display_class, sev = final_decision(pred, conf)

                st.audio(wav_path)
                st.image(spec, caption="Spectrogram")

                st.write(f"**Predicted:** {display_class}")
                if display_class == "Dysarthria":
                    st.write(f"**Confidence:** {conf:.2f}")
                st.write(f"**Severity:** {sev or 'None'}")

                st.session_state.last_severity = sev

                add_result(st.session_state.user, None, conf, sev or "Normal", "Upload", display_class, spec)
                
                gc.collect()

    # -------- Record --------
    else:
        dur = st.slider("Duration", 1, 5, 2)
        if st.button("Record now"):
            base = f"{st.session_state.user}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            wav = os.path.join("uploads", f"{base}.wav")
            record_audio_wav(dur, filename=wav)

            y = read_audio_from_any(wav)
            if y is not None:
                img, spec = audio_to_model_ready_image_file(y, base)
                pred, conf = predict_from_img_array(img)

                display_class, sev = final_decision(pred, conf)

                st.audio(wav)
                st.image(spec, caption="Spectrogram")

                st.write(f"**Predicted:** {display_class}")
                if display_class == "Dysarthria":
                    st.write(f"**Confidence:** {conf:.2f}")
                st.write(f"**Severity:** {sev or 'None'}")

                st.session_state.last_severity = sev

                add_result(st.session_state.user, None, conf, sev or "Normal", "Record", display_class, spec)
                
                gc.collect()

# ---------------- Therapy ----------------
with tabs[3]:
    st.subheader("📝 Therapy Exercises (auto up to 3 attempts)")
    if "last_severity" not in st.session_state:
        st.warning("⚠️ Please do assessment first")

    elif st.session_state.last_severity is None:
        st.success("🧠 Your current level: Normal")
        st.info("✅ No speech disorder detected. Therapy not required.")

    else:
        st.info(f"🧠 Your current level: {st.session_state.last_severity}")

    # ✅ Allow therapy only if severity exists
    if st.session_state.get("last_severity") is not None:
        if st.button("▶️ Start Therapy"):
            start_new_round()
            st.rerun()

    if st.session_state.therapy_active:
        target = st.session_state.therapy_word
        attempt = st.session_state.attempt
        st.info(f"🗣️ Speak this: **{target}** (Attempt {attempt}/3)")

        # ---- Speak the target ----
        speak_text(target)
        time.sleep(0.6)

        # ---- Capture response ----
        spoken = capture_speech(duration=3)
        st.write(f"Attempt {attempt} — You said: {spoken}")

        if normalize_phrase(spoken) == normalize_phrase(target):
            st.success(f"✅ GOOD — matched on attempt {attempt}")
            add_result(st.session_state.user, target, 1.0, "Good", "Exercise", target, "")
            st.session_state.therapy_active = False
        else:
            st.error(f"❌ BAD — attempt {attempt} failed")
            add_result(st.session_state.user, target, 0.0, "Bad", "Exercise", spoken or "Unrecognized", "")
            st.session_state.attempts_left -= 1

            if st.session_state.attempts_left > 0:
                st.session_state.attempt += 1
                st.info(f"Repeating... {st.session_state.attempts_left} attempts left")
                time.sleep(0.8)
                st.rerun()
            else:
                st.warning("⚠️ Attempts exhausted.")
                st.session_state.therapy_active = False
                
#-------------------pdf--------------
                
with tabs[4]:
    st.subheader("📄 Smart Report Generator")
    st.info("Generate a detailed PDF report of your speech progress.")
    
    start_date = st.date_input("Start Date")
    end_date = st.date_input("End Date")
    
    if st.button("📥 Generate Report"):

        all_data = fetch_results_for_user(st.session_state.user)
        data = []

        if start_date > end_date:
            st.error("Start date should be before End date")
            st.stop()

        for d in all_data:
            record_date = datetime.fromisoformat(d[0]).date()
            if start_date <= record_date <= end_date:
                data.append(d)

        if not data:
            st.warning("No data available to generate report.")
        else:
            rows = []
            for d in data:
                rows.append({
                    "Time": d[0],
                    "Prediction": d[1],
                    "Confidence": round(d[2], 3),
                    "Severity": d[3],
                    "Source": d[4]
                })

            df = pd.DataFrame(rows)
            st.success("✅ Data fetched successfully!")

            st.subheader("📋 Report Preview")
            st.dataframe(df)

            pdf_path = generate_pdf(df, st.session_state.user)

            with open(pdf_path, "rb") as f:
                st.download_button(
                    "📄 Download Report",
                    f,
                    file_name=pdf_path,
                    mime="application/pdf"
                )


with tabs[5]:
    # ─────────────────────────────────────────────
    # 📊 COMPARE TWO REPORTS
    # ─────────────────────────────────────────────
    st.markdown("---")
    st.subheader("📊 Compare Two Reports")
    st.info("Upload two previously downloaded PDF reports to see your progress comparison.")

    col1, col2 = st.columns(2)
    with col1:
        report1 = st.file_uploader("📄 Upload Earlier Report (Week 1)", type=["pdf"], key="report1")
    with col2:
        report2 = st.file_uploader("📄 Upload Recent Report (Week 2)", type=["pdf"], key="report2")

    def extract_table_from_report(pdf_file):
        import pdfplumber
        import io

        rows = []
        with pdfplumber.open(io.BytesIO(pdf_file.read())) as pdf:
            for page in pdf.pages:
                tables = page.extract_tables()
                for table in tables:
                    if not table or len(table) < 2:
                        continue
                    header = [str(h).strip() if h else "" for h in table[0]]
                    if "Confidence" not in header:
                        continue
                    for row in table[1:]:
                        if row and any(cell for cell in row):
                            rows.append(dict(zip(header, [str(c).strip() if c else "" for c in row])))
        return rows

    def parse_avg_confidence(rows):
        confidences = []
        for r in rows:
            try:
                confidences.append(float(r.get("Confidence", 0)))
            except:
                pass
        return confidences

    def severity_score(sev):
        mapping = {"Severe": 3, "Moderate": 2, "Mild": 1, "Normal": 0, "None": 0, "": 0}
        return mapping.get(sev, 0)

    if report1 and report2:
        with st.spinner("Analysing reports..."):
            rows1 = extract_table_from_report(report1)
            rows2 = extract_table_from_report(report2)

        if not rows1 or not rows2:
            st.error("❌ Could not extract data from one or both reports. Make sure you upload reports generated by this app.")
        else:
            confs1 = parse_avg_confidence(rows1)
            confs2 = parse_avg_confidence(rows2)

            avg1 = sum(confs1) / len(confs1) if confs1 else 0
            avg2 = sum(confs2) / len(confs2) if confs2 else 0

            sev1 = [r.get("Severity", "") for r in rows1]
            sev2 = [r.get("Severity", "") for r in rows2]
            avg_sev1 = sum(severity_score(s) for s in sev1) / len(sev1) if sev1 else 0
            avg_sev2 = sum(severity_score(s) for s in sev2) / len(sev2) if sev2 else 0

            # ── Side-by-side metrics ──
            st.markdown("### 📋 Summary Comparison")
            m1, m2 = st.columns(2)
            with m1:
                st.metric("📄 Earlier Report — Avg Confidence", f"{avg1:.3f}")
                st.metric("Tests in Earlier Report", len(rows1))
            with m2:
                delta_val = round(avg2 - avg1, 3)
                st.metric("📄 Recent Report — Avg Confidence", f"{avg2:.3f}",
                          delta=f"{delta_val:+.3f}")
                st.metric("Tests in Recent Report", len(rows2))

            # ── Overall verdict ──
            st.markdown("### 🧠 Progress Analysis")

            conf_diff = avg2 - avg1
            sev_diff  = avg_sev2 - avg_sev1

            if conf_diff < -0.05 and sev_diff <= 0:
                verdict = "📈 **Significant Improvement!** Your dysarthria confidence score dropped, meaning speech clarity has improved."
                verdict_color = "success"
            elif conf_diff < 0:
                verdict = "✅ **Mild Improvement.** Slight positive trend detected. Keep practising!"
                verdict_color = "success"
            elif conf_diff > 0.05:
                verdict = "⚠️ **Decline Detected.** Dysarthria confidence increased. More therapy sessions are recommended."
                verdict_color = "warning"
            elif abs(conf_diff) <= 0.05:
                verdict = "➡️ **Stable.** No significant change between the two reports."
                verdict_color = "info"
            else:
                verdict = "ℹ️ **Inconclusive.** More data needed for a clear trend."
                verdict_color = "info"

            if verdict_color == "success":
                st.success(verdict)
            elif verdict_color == "warning":
                st.warning(verdict)
            else:
                st.info(verdict)

            # ── Confidence trend chart ──
            st.markdown("### 📊 Confidence Distribution Comparison")

            fig, ax = plt.subplots(figsize=(8, 3))
            if confs1:
                ax.plot(range(len(confs1)), confs1, marker='o', label="Earlier Report", color="steelblue")
            if confs2:
                ax.plot(range(len(confs2)), confs2, marker='s', label="Recent Report", color="tomato")
            ax.set_xlabel("Test #")
            ax.set_ylabel("Confidence")
            ax.set_title("Confidence Trend: Earlier vs Recent")
            ax.legend()
            st.pyplot(fig)
            plt.close()

            # ── Severity breakdown ──
            st.markdown("### 🩺 Severity Breakdown")
            from collections import Counter

            sev_col1, sev_col2 = st.columns(2)
            with sev_col1:
                st.markdown("**Earlier Report**")
                c1 = Counter(sev1)
                st.dataframe(
                    pd.DataFrame(c1.items(), columns=["Severity", "Count"])
                    .sort_values("Count", ascending=False)
                    .reset_index(drop=True)
                )
            with sev_col2:
                st.markdown("**Recent Report**")
                c2 = Counter(sev2)
                st.dataframe(
                    pd.DataFrame(c2.items(), columns=["Severity", "Count"])
                    .sort_values("Count", ascending=False)
                    .reset_index(drop=True)
                )

            # ── Therapist tip ──
            st.markdown("### 💡 Therapist Tip")
            if "Improvement" in verdict:
                st.success("Great work! Continue your current therapy routine. Try increasing difficulty in exercises.")
            elif "Decline" in verdict:
                st.warning("Don't worry — consistency is key. Focus on slower, deliberate pronunciation exercises.")
            else:
                st.info("Maintain your routine. Try logging more sessions for better tracking accuracy.")
            
    
with tabs[6]:
    st.subheader("🤖 Virtual Therapist (AI Chatbot)")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            {"role": "assistant", "content": "Hello! I'm your speech therapist. How can I help you?"}
        ]

    # Display chat
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    user_input = st.chat_input("Type your message...")

    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})

        try:
            from groq import Groq

            client = Groq(api_key=st.secrets["GROQ_API_KEY"])

            severity = st.session_state.get("last_severity", "Unknown")

            messages = [
                {"role": "system", "content": f"""
You are a speech therapist.
Patient severity: {severity}.
Give short, simple, encouraging advice.
Do not diagnose.
"""}
            ]

            for msg in st.session_state.chat_history:
                messages.append(msg)

            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=messages,
                max_tokens=300
            )

            reply = response.choices[0].message.content

        except Exception as e:
            reply = f"Error: {e}"

        st.session_state.chat_history.append({"role": "assistant", "content": reply})
        st.rerun()