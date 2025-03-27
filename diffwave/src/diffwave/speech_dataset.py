import torchaudio as T

audio, sr = T.load('/data/Speech_Commands/SC09/one/856eb138_nohash_4.wav')
print(audio.shape, sr)