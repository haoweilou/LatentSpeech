scp dataset.py function.py model.py params.py pqmf.py trainvqae.py tts_config.py haoweilou@drstrange.cse.unsw.edu.au:~/LatentSpeech/
ssh haoweilou@drstrange.cse.unsw.edu.au
scp dataset.py function.py model.py params.py pqmf.py trainvqae.py tts_config.py test.py trainae.py haoweilou@drstrange.cse.unsw.edu.au:~/LatentSpeech/


scp haoweilou@drstrange.cse.unsw.edu.au:/home/haoweilou/LatentSpeech/model/vqae_100.pth ./model/
scp haoweilou@drstrange.cse.unsw.edu.au:/home/haoweilou/LatentSpeech/model/ae_400.pth ./model/

scp dataset.py trainae_2set haoweilou@drstrange.cse.unsw.edu.au:~/LatentSpeech/
# Copy Model
scp haoweilou@drstrange.cse.unsw.edu.au:/home/haoweilou/LatentSpeech/model/StyleSpeech_450.pth ./model/

# copy log
scp haoweilou@drstrange.cse.unsw.edu.au:/home/haoweilou/LatentSpeech/log/loss_vqaeinit ./log

# copy a file 
scp trainvqae_init.py haoweilou@drstrange.cse.unsw.edu.au:~/LatentSpeech/
scp trainvqae_init.py codebook model.py haoweilou@drstrange.cse.unsw.edu.au:~/LatentSpeech/

# Generate a codebook 
python visualize.py

scp haoweilou@drstrange.cse.unsw.edu.au:~/LatentSpeech/model/loud_600.pth  .\model
scp haoweilou@drstrange.cse.unsw.edu.au:~/LatentSpeech/model/wave_600.pth  .\model
scp haoweilou@drstrange.cse.unsw.edu.au:~/LatentSpeech/model/vqae1_600.pth  .\model

user id: hl6114
/g/data/ey69/hw
ssh hl6114@gadi.nci.org.au
scp hl6114@gadi.nci.org.au:/g/data/ey69/haowei

# Training Time Record
V100: 1.7s, two 12,1024 upsampler, 500+MB