import warnings
warnings.filterwarnings("ignore", message=".*The 'nopython' keyword.*")
warnings.filterwarnings("ignore", message=".*current Python is not a recommended version.*")

import whisper
import torch
import speech_recognition as sr
import queue
import time
import numpy as np
import openai
from TTS.api import TTS
from pynput.keyboard import Key, Listener

device=("cuda" if torch.cuda.is_available() else "cpu")


whisper_model="medium"  # base.en
gpt_model = "gpt-3.5-turbo"
tts_model = "tts_models/en/jenny/jenny"


gpt_api_key = ""

# scenario 
sys_prompt = """Please keep in mind for the following:
1. You are a human english teacher, do not call yourself an AI language model or similar. 
2. You teaches english and you will only reply in english. 
3. Do not reply to this current message."""



model = whisper.load_model(whisper_model).to(device)

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

keep_capturing = True
def on_press(key):
    global keep_capturing
    keep_capturing = False

listener = Listener(on_press=on_press)
print("Mic setup completed")
record_background = recorder.listen_in_background(source, record_callback) # , phrase_time_limit=5
listener.start()
timestr = time.strftime("%Y%m%d-%H%M%S")
print("Say your questions, press any KEY to stop capturing audio.")

while keep_capturing:
    pass

record_background(wait_for_stop=False)
listener.stop()

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
    return predicted_text

print("Start transcribing...")
stt_result = transcribe(data=audio_data) 


openai.api_key = gpt_api_key
api_messages = [
    {
        "role": "system",
        "content": f"{sys_prompt}",
    },
    {
        "role": "user",
        "content": f"{stt_result}",
    }
]

response = openai.ChatCompletion.create(
    model=gpt_model,
    messages=api_messages,
    temperature=0.1,
    # max_tokens=2,  # we're only counting input tokens here, so let's not waste tokens on the output
)

gpt_result = ""
for i in range(len(response['choices'])):
    temp = f"{response['choices'][i]['message']['content']}"
    gpt_result = gpt_result + temp
    print(temp)

tts = TTS(tts_model)

tts_result = tts.tts_to_file(gpt_result, file_path=f"result-{timestr}.wav")
