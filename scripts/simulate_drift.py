import os
import time
import requests
import glob

API_URL = "http://localhost:8000/predict"

# Find a real audio file from your repository dataset to use as a valid payload
search_path = os.path.join("data", "genres_original", "**", "*.wav")
audio_files = glob.glob(search_path, recursive=True)

if not audio_files:
    print("⚠️ Could not find real .wav files inside 'data/genres_original/'.")
    print("Fallback: Please ensure a small valid audio file exists to test the pipeline.")
    # Create an absolute fallback reference if needed
    exit(1)

# Grab the first available valid audio file path
sample_audio_path = audio_files[0]
print(f"🎵 Using real audio file for simulation payload: {sample_audio_path}")

print("🚀 Starting Production Data Simulation...")

def send_real_request():
    try:
        with open(sample_audio_path, "rb") as f:
            audio_bytes = f.read()
        
        # NOTE: Change "file" to "audio" if your FastAPI parameter is named audio
        payload_files = {"file": (os.path.basename(sample_audio_path), audio_bytes, "audio/wav")}
        
        response = requests.post(API_URL, files=payload_files, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Status: 200 | Prediction: {data.get('genre_prediction', 'N/A')} | p-value: {data.get('statistical_drift_p_value', 1.0):.4f}")
        else:
            print(f"⚠️ Server returned status code: {response.status_code}")
            print(f"   Details: {response.text}")
            
    except Exception as e:
        print(f"❌ Request failed: {e}")

# Keep sending data streams every second to populate the Grafana dashboard metrics
try:
    while True:
        send_real_request()
        time.sleep(1.0)
except KeyboardInterrupt:
    print("\n🛑 Traffic simulation stopped safely.")