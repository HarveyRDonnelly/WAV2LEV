import numpy as np
from copy import deepcopy
from utils.urgent_helpers import *
from utils.rir_utils import estimate_early_rir
import pandas as pd
import soundfile as sf
from scipy.signal import resample


MAX_RIR_SAMPLE = 2.0


def apply_noise(fg_audio, bg_audio, row):
    fs = row['fs']
    rng = np.random.default_rng(row['seed'])
    
    if fg_audio.ndim == 1:
        speech_sample = fg_audio[None, :]
    else:
        speech_sample = fg_audio
    
    if bg_audio.ndim == 1:
        noise_sample = bg_audio[None, :]
    else:
        noise_sample = bg_audio
    
    noisy_speech = deepcopy(speech_sample)
    
    if 'rir_sample' in row and not pd.isnull(row['rir_sample']):
        rir_file = row['rir_sample']
        rir_sample, rir_fs = sf.read(rir_file)
        
        if rir_fs != fs:
            num_samples = int(len(rir_sample) * fs / rir_fs)
            rir_sample = resample(rir_sample, num_samples)
            
        if rir_sample.ndim > 1:
            rir_sample = rir_sample[:, 0] if rir_sample.shape[1] > 1 else rir_sample.squeeze()
            
        if rir_sample.ndim == 1:
            rir_sample = rir_sample[None, :]
            
        max_rir_length = int(MAX_RIR_SAMPLE * fs)  # 1 second max
        if rir_sample.shape[-1] > max_rir_length:
            rir_sample = rir_sample[:, :max_rir_length]
            
        noisy_speech = add_reverberation(noisy_speech, rir_sample)
        early_rir_sample = estimate_early_rir(rir_sample, fs=fs)
        speech_sample = add_reverberation(speech_sample, early_rir_sample)
    
    # Mix noise
    noisy_speech, noise_mixed = mix_noise(
        noisy_speech, noise_sample, snr=row['snr_db'], rng=rng
    )
    
    # Apply bandwidth limitation
    noisy_speech = bandwidth_limitation(
        noisy_speech, fs=fs, fs_new=row['fs_new'], res_type="kaiser_best"
    )
    
    # Apply clipping
    noisy_speech = clipping(
        noisy_speech, 
        min_quantile=row['clip_min_quantile'], 
        max_quantile=row['clip_max_quantile']
    )
    
    # Apply codec compression
    noisy_speech = codec_compression(
        noisy_speech, 
        fs, 
        format=row['codec_format'], 
        encoder=row['codec_encoder'], 
        qscale=row['codec_qscale']
    )
    
    # Apply packet loss
    noisy_speech = packet_loss(
        noisy_speech, 
        fs, 
        packet_loss_indices=row['packet_loss_indices'], 
        packet_duration_ms=20
    )
    
    # Normalize
    scale = 0.9 / max(
        np.max(np.abs(noisy_speech)),
        np.max(np.abs(speech_sample)),
        np.max(np.abs(noise_mixed)),
    )
    
    return (noisy_speech * scale).squeeze()