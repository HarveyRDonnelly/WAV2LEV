from utils.noising import apply_noise
import torchaudio, torch

def generate_cnoisy_audio(row):
    
    fg_audio = get_fg_audio(row)
    bg_audio = get_bg_audio(row)
    
    cnoisy_audio = apply_noise(
        fg_audio,
        bg_audio,
        row
    )
    
    return cnoisy_audio

def _load_wav_mono_16k(path, max_samples=None):
    wav, sr = torchaudio.load(path)
    if wav.size(0) > 1:
        wav = torch.mean(wav, dim=0, keepdim=True)
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
    wav = wav.squeeze(0)
    if max_samples is not None and wav.numel() > max_samples:
        wav = wav[:max_samples]
    return wav

def get_bg_audio(row, max_samples=None):
    return _load_wav_mono_16k(row['bg_wav_path'], max_samples)

def get_fg_audio(row, max_samples=None):
    return _load_wav_mono_16k(row['fg_wav_path'], max_samples)

def get_cnoisy_audio(row, max_samples=None):
    wav, sr = torchaudio.load(row['cnoisy_wav_path'])
    if wav.size(0) > 1:
        wav = torch.mean(wav, dim=0, keepdim=True)
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
    wav = wav.squeeze(0)
    if max_samples is not None and wav.numel() > max_samples:
        wav = wav[:max_samples]
    return wav
