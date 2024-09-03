from dataset import BakerAudio, LJSpeechAudio
from function import save_audio
baker_dataset = BakerAudio(0,100,"L:/baker/")
lj_dataset = LJSpeechAudio(0,10000,"L:/LJSpeech/")

baker_audio = baker_dataset.audios[0].unsqueeze(0).unsqueeze(0)
save_audio(baker_audio[0].detach().cpu(),48000,"real_baker")

lj_audio = lj_dataset.audios[0].unsqueeze(0).unsqueeze(0)
save_audio(lj_audio[0].detach().cpu(),48000,"real_lj")
