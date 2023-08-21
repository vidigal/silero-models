import os
import torch
import soundfile as sf
import numpy as np

device = torch.device('cpu')
torch.set_num_threads = 14

model_v3_en_indic_path = "./models/v3_en_indic.pt"
model_v3_en_path = "./models/v3_en.pt"
output_path = "./output/"
output_file_name = "1k-you-will-win-in-life.wav"

principal_speaker = "en_84"

sample_rate = 48000
amount_of_times = 1000

starter_text = "<speak><prosody rate='slow' pitch='x-high'>Subscribe to request your own mantra by making a comment</prosody><break time='1s' strength='x-weak'/></speak>"
text_to_speach = "<speak><prosody rate='x-slow'>You will win in life!</prosody></speak>"


model_v3_en_indic = torch.package.PackageImporter(model_v3_en_indic_path).load_pickle("tts_models", "model")
model_v3_en_indic.to(device)

model_v3_en = torch.package.PackageImporter(model_v3_en_path).load_pickle("tts_models", "model")
model_v3_en.to(device)


audios = []

# Insere a frase inicial
audios.append(model_v3_en.apply_tts(ssml_text=starter_text, speaker=principal_speaker, sample_rate=sample_rate))

step = 1
for i in range(amount_of_times):
    print(f"Step: {step}")
    step += 1

    if i % 2 == 0:
        audios.append(model_v3_en.apply_tts(ssml_text=text_to_speach, speaker="random", sample_rate=sample_rate))
    else:
        audios.append(model_v3_en_indic.apply_tts(ssml_text=text_to_speach, speaker="random", sample_rate=sample_rate))

audio_final = np.concatenate(audios)
sf.write(f"{output_path}{output_file_name}", audio_final, sample_rate)

# for speaker in model.speakers:
#     print(speaker)
#     audio = model.apply_tts(text=text_to_speach, speaker=speaker, sample_rate=sample_rate)
#     sf.write(f"{speaker}.wav", audio, sample_rate)
