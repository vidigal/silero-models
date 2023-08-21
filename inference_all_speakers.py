import os
import torch
import soundfile as sf
import numpy as np

device = torch.device('cpu')
torch.set_num_threads = 16

model_v3_en_path = "./models/v3_en.pt"
output_path = "./output/"


sample_rate = 48000

text_to_speach = "<speak><prosody rate='slow' pitch='x-high'>Subscribe to request your own mantra by making a comment</prosody><break time='1s' strength='x-weak'/></speak>"
 
model_v3_en = torch.package.PackageImporter(model_v3_en_path).load_pickle("tts_models", "model")
model_v3_en.to(device)


for speaker in model_v3_en.speakers:
    print(speaker)
    audio = model_v3_en.apply_tts(ssml_text=text_to_speach, speaker=speaker, sample_rate=sample_rate)
    sf.write(f"{output_path}{speaker}.wav", audio, sample_rate)

