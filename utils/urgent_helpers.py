"""
HELPERS COPIED FROM: https://github.com/urgent-challenge/urgent2025_challenge/
"""

import subprocess
from pathlib import Path
import requests
import librosa
import numpy as np
import scipy
import soundfile as sf
import sys
import pandas as pd
import ast
import torch
from torchaudio.io import AudioEffector, CodecConfig

ffmpeg = "/usr/bin/ffmpeg"


"""
SOURCED FROM ESPNET: https://github.com/espnet/espnet/blob/master/espnet2/train/preprocessor.py
"""
def framing(
    x,
    frame_length: int = 512,
    frame_shift: int = 256,
    centered: bool = True,
    padded: bool = True,
):
    if x.size == 0:
        raise ValueError("Input array size is zero")
    if frame_length < 1:
        raise ValueError("frame_length must be a positive integer")
    if frame_length > x.shape[-1]:
        raise ValueError("frame_length is greater than input length")
    if 0 >= frame_shift:
        raise ValueError("frame_shift must be greater than 0")

    if centered:
        pad_shape = [(0, 0) for _ in range(x.ndim - 1)] + [
            (frame_length // 2, frame_length // 2)
        ]
        x = np.pad(x, pad_shape, mode="constant", constant_values=0)

    if padded:
        # Pad to integer number of windowed segments
        # I.e make x.shape[-1] = frame_length + (nseg-1)*nstep,
        #  with integer nseg
        nadd = (-(x.shape[-1] - frame_length) % frame_shift) % frame_length
        pad_shape = [(0, 0) for _ in range(x.ndim - 1)] + [(0, nadd)]
        x = np.pad(x, pad_shape, mode="constant", constant_values=0)

    # Created strided array of data segments
    if frame_length == 1 and frame_length == frame_shift:
        result = x[..., None]
    else:
        shape = x.shape[:-1] + (
            (x.shape[-1] - frame_length) // frame_shift + 1,
            frame_length,
        )
        strides = x.strides[:-1] + (frame_shift * x.strides[-1], x.strides[-1])
        result = np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)
    return result

"""
SOURCED FROM ESPNET: https://github.com/espnet/espnet/blob/master/espnet2/train/preprocessor.py
"""
def detect_non_silence(
    x: np.ndarray,
    threshold: float = 0.01,
    frame_length: int = 1024,
    frame_shift: int = 512,
    window: str = "boxcar",
) -> np.ndarray:
    """Power based voice activity detection.

    Args:
        x: (Channel, Time)
    >>> x = np.random.randn(1000)
    >>> detect = detect_non_silence(x)
    >>> assert x.shape == detect.shape
    >>> assert detect.dtype == np.bool
    """
    if x.shape[-1] < frame_length:
        return np.full(x.shape, fill_value=True, dtype=np.bool)

    if x.dtype.kind == "i":
        x = x.astype(np.float64)
    # framed_w: (C, T, F)
    framed_w = framing(
        x,
        frame_length=frame_length,
        frame_shift=frame_shift,
        centered=False,
        padded=True,
    )
    framed_w *= scipy.signal.get_window(window, frame_length).astype(framed_w.dtype)
    # power: (C, T)
    power = (framed_w**2).mean(axis=-1)
    # mean_power: (C, 1)
    mean_power = np.mean(power, axis=-1, keepdims=True)
    if np.all(mean_power == 0):
        return np.full(x.shape, fill_value=True, dtype=np.bool)
    # detect_frames: (C, T)
    detect_frames = power / mean_power > threshold
    # detects: (C, T, F)
    detects = np.broadcast_to(
        detect_frames[..., None], detect_frames.shape + (frame_shift,)
    )
    # detects: (C, TF)
    detects = detects.reshape(*detect_frames.shape[:-1], -1)
    # detects: (C, TF)
    return np.pad(
        detects,
        [(0, 0)] * (x.ndim - 1) + [(0, x.shape[-1] - detects.shape[-1])],
        mode="edge",
    )

def buildFFmpegCommand(params):

    filter_commands = ""
    filter_commands += "[1:a]asplit=2[sc][mix];"
    filter_commands += (
        "[0:a][sc]sidechaincompress="
        + f"threshold={params['threshold']}:"
        + f"ratio={params['ratio']}:"
        + f"level_sc={params['sc_gain']}"
        + f":release={params['release']}"
        + f":attack={params['attack']}"
        + "[compr];"
    )
    filter_commands += "[compr][mix]amix"

    commands_list = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "quiet",
        "-i",
        params["speech_path"],
        "-i",
        params["noise_path"],
        "-filter_complex",
        filter_commands,
        params["output_path"],
    ]

    return commands_list


#############################
# Augmentations per sample
#############################
def mix_noise(speech_sample, noise_sample, snr=5.0, rng=None):
    """Mix the speech sample with an additive noise sample at a given SNR.

    Args:
        speech_sample (np.ndarray): a single speech sample (Channel, Time)
        noise_sample (np.ndarray): a single noise sample (Channel, Time)
        snr (float): signal-to-nosie ratio (SNR) in dB
        rng (np.random.Generator): random number generator
    Returns:
        noisy_sample (np.ndarray): output noisy sample (Channel, Time)
        noise (np.ndarray): scaled noise sample (Channel, Time)
    """
    len_speech = speech_sample.shape[-1]
    len_noise = noise_sample.shape[-1]
    if len_noise < len_speech:
        offset = rng.integers(0, len_speech - len_noise)
        # Repeat noise
        noise_sample = np.pad(
            noise_sample,
            [(0, 0), (offset, len_speech - len_noise - offset)],
            mode="wrap",
        )
    elif len_noise > len_speech:
        offset = rng.integers(0, len_noise - len_speech)
        noise_sample = noise_sample[:, offset : offset + len_speech]

    power_speech = (speech_sample[detect_non_silence(speech_sample)] ** 2).mean()
    power_noise = (noise_sample[detect_non_silence(noise_sample)] ** 2).mean()
    scale = 10 ** (-snr / 20) * np.sqrt(power_speech) / np.sqrt(max(power_noise, 1e-10))
    noise = scale * noise_sample
    noisy_speech = speech_sample + noise
    return noisy_speech, noise

def add_reverberation(speech_sample, rir_sample):
    """Mix the speech sample with an additive noise sample at a given SNR.

    Args:
        speech_sample (np.ndarray): a single speech sample (1, Time)
        rir_sample (np.ndarray): a single room impulse response (RIR) (Channel, Time)
    Returns:
        reverberant_sample (np.ndarray): output noisy sample (Channel, Time)
    """
    reverberant_sample = scipy.signal.convolve(speech_sample, rir_sample, mode="full")
    return reverberant_sample[:, : speech_sample.shape[1]]


def bandwidth_limitation(speech_sample, fs: int, fs_new: int, res_type="kaiser_best"):
    """Apply the bandwidth limitation distortion to the input signal.

    Args:
        speech_sample (np.ndarray): a single speech sample (1, Time)
        fs (int): sampling rate in Hz
        fs_new (int): effective sampling rate in Hz
        res_type (str): resampling method

    Returns:
        ret (np.ndarray): bandwidth-limited speech sample (1, Time)
    """
    opts = {"res_type": res_type}
    if fs == fs_new:
        return speech_sample
    assert fs > fs_new, (fs, fs_new)
    ret = librosa.resample(speech_sample, orig_sr=fs, target_sr=fs_new, **opts)
    # resample back to the original sampling rate
    ret = librosa.resample(ret, orig_sr=fs_new, target_sr=fs, **opts)
    return ret[:, : speech_sample.shape[1]]


def clipping(speech_sample, min_quantile: float = 0.0, max_quantile: float = 0.9):
    """Apply the clipping distortion to the input signal.

    Args:
        speech_sample (np.ndarray): a single speech sample (1, Time)
        min_quantile (float): lower bound on the quantile of samples to be clipped
        max_quantile (float): upper bound on the quantile of samples to be clipped

    Returns:
        ret (np.ndarray): clipped speech sample (1, Time)
    """
    q = np.array([min_quantile, max_quantile])
    min_, max_ = np.quantile(speech_sample, q, axis=-1, keepdims=False)
    # per-channel clipping
    ret = np.stack(
        [
            np.clip(speech_sample[i], min_[i], max_[i])
            for i in range(speech_sample.shape[0])
        ],
        axis=0,
    )
    return ret


"""
def codec_compression(speech_sample, fs: int, vbr_quality: float):
    # if random.random() > 0.5:
    #     module = Pedalboard([GSMFullRateCompressor()])
    # else:
    #     module = Pedalboard([MP3Compressor()])
    # vbr_quality = random.uniform(
    #     params["mp3_vbr_quality"][0], params["mp3_vbr_quality"][1]
    # )
    # print(vbr_quality)
    assert 0.0 <= vbr_quality <= 10.0
    module = Pedalboard([MP3Compressor(vbr_quality=vbr_quality)])
    output = module(speech_sample, fs)
    return output
"""


def codec_compression(
    speech_sample,
    fs: int,
    format: str,
    encoder: str = None,
    qscale: int = None,
):
    
    encoder = None if pd.isnull(encoder) else encoder
    assert format in ["mp3", "ogg"], format
    assert encoder in [None, "None", "vorbis", "opus"], encoder
    
    # Validate format-encoder combinations
    if format == "mp3":
        if encoder not in [None, "None"]:
            raise ValueError(f"MP3 format only supports default encoder, got: {encoder}")
        encoder = None  # Ensure encoder is None for MP3
    elif format == "ogg":
        if encoder in [None, "None"]:
            encoder = "vorbis"  # Default to vorbis for ogg
        elif encoder not in ["vorbis", "opus"]:
            raise ValueError(f"Ogg format only supports 'vorbis' or 'opus' encoders, got: {encoder}")

    encoder = None if encoder == "None" else encoder
    
    if speech_sample.ndim == 2:
        speech_sample = speech_sample.T  # (channel, sample) -> (sample, channel)
    
    try:
        module = AudioEffector(
            format=format,
            encoder=encoder,
            codec_config=CodecConfig(qscale=qscale),
            pad_end=True,
        )
        output = module.apply(torch.from_numpy(speech_sample), fs).numpy()
    except Exception as e:
        print(f"Error with format={format}, encoder={encoder}, qscale={qscale}", flush=True)
        print(e, flush=True)
        raise

    if output.shape[0] < speech_sample.shape[0]:
        zeros = np.zeros((speech_sample.shape[0] - output.shape[0], output.shape[1]))
        output = np.concatenate((output, zeros), axis=0)
    elif output.shape[0] > speech_sample.shape[0]:
        output = output[: speech_sample.shape[0]]

    assert speech_sample.shape == output.shape, (speech_sample.shape, output.shape)
    return (
        output.T if output.ndim == 2 else output
    )  # (sample, channel) -> (channel, sample)



def packet_loss(
    speech_sample, fs: int, packet_loss_indices: list, packet_duration_ms: int = 20
):
    for idx in ast.literal_eval(packet_loss_indices):
        start = idx * packet_duration_ms * fs // 1000
        end = (idx + 1) * packet_duration_ms * fs // 1000
        speech_sample[:, start:end] = 0

    return speech_sample
