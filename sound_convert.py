import os
import gc
import librosa
import numpy as np
import matplotlib.pyplot as plt

# Ensure the 'dataset' folder and subfolders are created
def create_data_folders(categories, base_dir="dataset"):
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    for category in categories:
        category_path = os.path.join(base_dir, category)
        if not os.path.exists(category_path):
            os.makedirs(category_path)

# Initialize categories and data directories
DATADIR = "data"
DATADIR1 = "dataset"
CATEGORIES = [class_name for class_name in os.listdir(DATADIR)]

# Create dataset folders
create_data_folders(CATEGORIES, DATADIR1)

# Configuration for audio processing
class conf:
    sampling_rate = 44100
    duration = 1
    hop_length = 350 * duration
    fmin = 1
    fmax = sampling_rate // 2
    n_mels = 256
    n_fft = n_mels * 20
    samples = sampling_rate * duration

# Read audio safely
def read_audio(conf, pathname, trim_long_data):
    try:
        y, sr = librosa.load(pathname, sr=conf.sampling_rate)
    except Exception as e:
        print(f"❌ Could not load {pathname}: {e}")
        return None

    if y is None or len(y) == 0:
        return None

    # Trim silence
    y, _ = librosa.effects.trim(y)

    # Pad or trim to fixed length
    if len(y) > conf.samples:
        if trim_long_data:
            y = y[:conf.samples]
    else:
        padding = conf.samples - len(y)
        offset = padding // 2
        y = np.pad(y, (offset, conf.samples - len(y) - offset), 'constant')
    return y

# Convert audio to mel-spectrogram
def audio_to_melspectrogram(conf, audio):
    spectrogram = librosa.feature.melspectrogram(
        y=audio,
        sr=conf.sampling_rate,
        n_mels=conf.n_mels,
        hop_length=conf.hop_length,
        n_fft=conf.n_fft,
        fmin=conf.fmin,
        fmax=conf.fmax
    )
    spectrogram = librosa.power_to_db(spectrogram)
    return spectrogram.astype(np.float32)

# Convert file to mel-spectrogram
def read_as_melspectrogram(conf, pathname, trim_long_data):
    x = read_audio(conf, pathname, trim_long_data)
    if x is None:
        return None
    return audio_to_melspectrogram(conf, x)

# Main loop
lp = 0
for category in CATEGORIES:
    path = os.path.join(DATADIR, category)
    path1 = os.path.join(DATADIR1, category)

    for i, fn in enumerate(os.listdir(path)):
        pathi = os.path.join(path, fn)
        print(f"Processing {i}: {fn}")

        # Convert to mel-spectrogram
        x = read_as_melspectrogram(conf, pathi, trim_long_data=False)
        if x is None:
            print(f"⚠️ Skipping {fn}")
            continue

        lp += 1
        filename = str(lp)  # Sequential filename

        # Save spectrogram images (two copies as in your code)
        plt.imshow(x, interpolation='nearest', aspect='auto', origin='lower')
        for ju in range(2):
            plt.savefig(os.path.join(path1, f"{ju}_{filename}.jpg"))
        plt.close()

        # Free memory
        del x
        gc.collect()

print("✅ Processing complete!")
