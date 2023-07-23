import warnings
warnings.filterwarnings("ignore", message=".*The 'nopython' keyword.*")

import whisper
import torch
import speech_recognition as sr
import queue
import time
import numpy as np
from pynput.keyboard import Key, Listener


device=("cuda" if torch.cuda.is_available() else "cpu")



whisper_model="medium"  # base.en



print("Loading whisper model...")
model = whisper.load_model(whisper_model).to(device)
print("Whisper model loaded.")

print("Setting up mic for audio source...")
# setup mic for audio source
audio_queue = queue.Queue()
result_queue = queue.Queue()
break_threads = False
mic_active = False

source = sr.Microphone(sample_rate=16000)
recorder = sr.Recognizer()
recorder.energy_threshold = 300
recorder.pause_threshold = 0.8
recorder.dynamic_energy_threshold = False
with source:
    recorder.adjust_for_ambient_noise(source)

def record_callback(_, audio:sr.AudioData):
    data = audio.get_raw_data()
    print(f"audioData: {len(data)}")
    audio_queue.put_nowait(data)

recorder_placeholder = None
print("Audio source ready.")

def get_all_audio(max_time=30):
    audio = bytes()
    time_start = time.time()
    print(audio_queue.empty())
    while not audio_queue.empty():
        cur_time = time.time() - time_start 
        print(cur_time)
        if cur_time > max_time:
            break
        audio += audio_queue.get()

    data = sr.AudioData(audio,16000,2)
    data = data.get_raw_data()
    return data

def preprocess(data):
    return torch.from_numpy(np.frombuffer(data, np.int16).flatten().astype(np.float32) / 32768.0)

def transcribe(data=None,realtime=False):
    if data is None:
        audio_data = get_all_audio()
    else:
        audio_data = data
    audio_data = preprocess(audio_data)
    result = model.transcribe(audio_data, fp16=False, language='english') # fp16=False

    predicted_text = result["text"]
    print(predicted_text)
    result_queue.put_nowait(predicted_text)
    print("Done.")

def question_asked():
    global audio_queue
    print("Retereving audio from queue...")
    audio_data = get_all_audio(10)
    print("Start transcribing...")
    transcribe(data=audio_data) 

print("Setting up keyboard listener...")
# Setup keybaord listener
listening = True
pressed = False
def on_press(key):
    global pressed
    global recorder_placeholder
    # print('{0} pressed'.format(key))
    if key == Key.space:
        if not pressed:
            pressed = True
            record_background = recorder.listen_in_background(source, record_callback)
            recorder_placeholder = record_background


def on_release(key):
    global recorder
    global pressed
    global listening
    global recorder_placeholder
    # print('{0} release'.format(key))
    if key == Key.space:
        pressed = False
        print("Stop capturing audio...")
        recorder_placeholder(wait_for_stop=False)
        question_asked()
    elif key == Key.esc:
        listening = False

listener = Listener(on_press=on_press, on_release=on_release)
print("Keyboard listener ready.")

# run listener in background so that the while loop gets executed
listener.start()
print("Mic setup complete, hold 'space' to talk")
while listening:
    # print("running some code")
    pass

listener.stop()

