
from TTS.api import TTS

# print(TTS.list_models())
# en_models = [
#     'tts_models/en/ek1/tacotron2', 
#     'tts_models/en/ljspeech/tacotron2-DDC', 
#     'tts_models/en/ljspeech/tacotron2-DDC_ph', 
#     'tts_models/en/ljspeech/glow-tts', 
#     'tts_models/en/ljspeech/speedy-speech', 
#     'tts_models/en/ljspeech/tacotron2-DCA', 
#     'tts_models/en/ljspeech/vits', 
#     'tts_models/en/ljspeech/vits--neon', 
#     'tts_models/en/ljspeech/fast_pitch', 
#     'tts_models/en/ljspeech/overflow', 
#     'tts_models/en/ljspeech/neural_hmm', 
#     'tts_models/en/vctk/vits', 
#     'tts_models/en/vctk/fast_pitch', 
#     'tts_models/en/sam/tacotron-DDC', 
#     'tts_models/en/blizzard2013/capacitron-t2-c50', 
#     'tts_models/en/blizzard2013/capacitron-t2-c150_v2', 
#     'tts_models/en/multi-dataset/tortoise-v2', 
#     'tts_models/en/jenny/jenny'
# ]


tts = TTS("tts_models/en/jenny/jenny")

print("Converting Text to speech into wav file...")
wav = tts.tts_to_file("As an English teacher, I don't have a favorite pet. However, some popular choices for pets include dogs, cats, birds, and fish. Each pet has its own unique qualities and can bring joy and companionship to their owners. Do you have a favorite pet?", file_path="output.wav")


