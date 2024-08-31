scp dataset.py function.py model.py params.py pqmf.py trainvqae.py tts_config.py haoweilou@drstrange.cse.unsw.edu.au:~/LatentSpeech/
ssh haoweilou@drstrange.cse.unsw.edu.au
scp dataset.py function.py model.py params.py pqmf.py trainvqae.py tts_config.py test.py trainae.py haoweilou@drstrange.cse.unsw.edu.au:~/LatentSpeech/


scp haoweilou@drstrange.cse.unsw.edu.au:/home/haoweilou/LatentSpeech/model/vqae_100.pth ./model/
scp haoweilou@drstrange.cse.unsw.edu.au:/home/haoweilou/LatentSpeech/model/ae_50.pth ./model/

scp haoweilou@drstrange.cse.unsw.edu.au:/home/haoweilou/scratch/model/TTS_hidden/Alinger/tts_600.pth ./save/model
scp haoweilou@drstrange.cse.unsw.edu.au:/home/haoweilou/scratch/model/TTS_hidden/Alinger/tts_wo_da_600.pth ./save/model
scp haoweilou@drstrange.cse.unsw.edu.au:/home/haoweilou/scratch/model/TTS_hidden/Alinger/tts_Melspec_600.pth ./save/model
scp haoweilou@drstrange.cse.unsw.edu.au:/home/haoweilou/scratch/model/TTS_hidden/Alinger/tts_MFCC_600.pth ./save/model

scp -r L:\baker\MFCC_Aligner haoweilou@drstrange.cse.unsw.edu.au:/home/haoweilou/scratch/baker

/home/haoweilou/scratch/model/TTS_hidden/Alinger 

iter_run.py to pretrain the aligner
log.txt is the log for aligner