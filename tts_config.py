word_dim = 256
# word_dim = 16
fft_config = {
    "head_num":2,
    "hidden_dim":word_dim,
    "filter_num":1024,
    "kernel_size":9,
    "dropout":0.1
}
# "word_num":88,#raw
# "word_num":46,#normalize
pho_config = {
    "word_num":88,
    "word_dim":word_dim,
    "padding_idx":0,
    "n_layers":4,
    "FFT":fft_config
}

style_config = {
    "word_num":8,
    "word_dim":word_dim,
    "padding_idx":0,
    "n_layers":4,   
    "FFT":fft_config
}

len_config = {
    "filter_num":256,
    "kernel_size":3,
    "dropout":0.5
}

fuse_config = {
    "word_dim":word_dim,
    "padding_idx":0,
    "n_layers":4,   
    "FFT":fft_config
}

config = {    
    "max_seq_len":512,
    "max_phone_len":128,
    "n_mel_channels":64,
    "pho_config":pho_config,
    "style_config":pho_config,
    "len_config":len_config,
    "fuse_config":fuse_config
}