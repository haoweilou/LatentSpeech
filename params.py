import numpy as np


class AttrDict(dict):
  def __init__(self, *args, **kwargs):
      super(AttrDict, self).__init__(*args, **kwargs)
      self.__dict__ = self

  def override(self, attrs):
    if isinstance(attrs, dict):
      self.__dict__.update(**attrs)
    elif isinstance(attrs, (list, tuple, set)):
      for attr in attrs:
        self.override(attr)
    elif attrs is not None:
      raise NotImplementedError
    return self


params = AttrDict(
    # Training params
    batch_size=32,
    learning_rate=5e-4,
    max_grad_norm=None,

    # Data params
    sample_rate=48*1000,
    n_mels=80,
    n_fft=1024,
    hop_samples=256,
    crop_mel_frames=12,  # Probably an error in paper.

    # Model params
    residual_layers=30,
    residual_channels=64,
    dilation_cycle_length=10,
    unconditional = False,
    noise_schedule=np.linspace(1e-4, 0.05, 50).tolist(),
    inference_noise_schedule=[0.0001, 0.001, 0.01, 0.05, 0.2, 0.5],

    # unconditional sample len
    audio_len = 3200*9, # unconditional_synthesis_samples
    max_time = 9,
    max_step = 50,

    #VAE
    # stft_scales = [2048, 1024, 512, 256, 128],
    stft_scales = [512, 256, 128, 64, 32],
    # stft_scales= [128, 64, 32,16],s
    log_epsilon = 1e-7,
    ratio = [4, 4, 2],
    # n_band = 4,
    n_band = 16,
    hidden_ch = 64,
    latent_size = 128
)