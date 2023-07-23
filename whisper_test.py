import warnings
warnings.filterwarnings("ignore", message=".*The 'nopython' keyword.*")

import whisper
import torch
import speech_recognition as sr
import queue
import time
import numpy as np

device=("cuda" if torch.cuda.is_available() else "cpu")


testModel="medium"  # base.en
testFile=None       # Use "None" to use your mic
# testFile="./audio_samples/sample-0.mp3"
# testFile="./audio_samples/sample-2.mp3"
# testFile="./audio_samples/sample-6.mp3"


model = whisper.load_model(testModel).to(device)

# decode_opts = whisper.DecodingOptions()
# , language="en"
if testFile is not None:
    result = model.transcribe(testFile)
    print(result["text"])
else:
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
        audio_queue.put_nowait(data)
    
    recorder.listen_in_background(source, record_callback, phrase_time_limit=2)
    print("Mic setup complete, you can now talk")

    def get_all_audio(min_time=-1):
        audio = bytes()
        got_audio = False
        time_start = time.time()
        while not got_audio or time.time() - time_start < min_time:
            while not audio_queue.empty():
                audio += audio_queue.get()
                got_audio = True

        data = sr.AudioData(audio,16000,2)
        data = data.get_raw_data()
        return data
    
    audio_data = get_all_audio()

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

    print("Start transcribing...")
    transcribe(data=audio_data) 
