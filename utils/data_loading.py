import torch
import numpy as np
import ast, sys
import torchaudio
from utils.audio_retrieval import get_cnoisy_audio
import whisper
from torch.nn.utils.rnn import pad_sequence
from utils.config import Config
import os

c = Config()

MAX_U = 1000

def collate_fn_cnoisy_precomputed(batch):
    vocab = ['<start>', 'ins', 'sub', 'del', 'match', '<end>']
    char2idx = {c: i for i, c in enumerate(vocab)}
    pad_token_id = -1
    model_id = "openai/whisper-large-v3"
    max_u = 400
    save_dir = c.EMBEDDINGS_PRECOMP_PATH

    audio_hiddens = []
    text_hiddens = []
    uncertainty_feats = []
    labels = []
    label_lengths = []
    real_wers = []
    speaker_ids = []
    asr_texts = []

    for item in batch:
        fg_id = item['fg_id']
        path = os.path.join(save_dir, f"{fg_id}.pt")
        data = torch.load(path)
        
        audio_hiddens.append(data['audio_hidden'])
        text_hiddens.append(data['text_hidden'])
        uncertainty_feats.append(data['uncertainty_feats'])
        asr_texts.append(data.get('asr_text', ""))
        
        original_seq = ast.literal_eval(item[f'asr_cnoisy_fg_multiclass_lev_{model_id}'])
        core_indices = [char2idx.get(c, char2idx['match']) for c in original_seq if c in ['ins','sub','del','match']]
        seq_indices = [char2idx['<start>']] + core_indices + [char2idx['<end>']]
        if len(seq_indices) > max_u:
            seq_indices = seq_indices[:max_u]
        
        labels.append(torch.tensor(seq_indices, dtype=torch.long))
        label_lengths.append(len(seq_indices))
        
        real_wers.append(max(0.0, min(1.0, item.get(f'asr_cnoisy_fg_wer_{model_id}', 0.0))))
        speaker_ids.append(fg_id)

    batch_audio = torch.stack(audio_hiddens)
    batch_text = pad_sequence(text_hiddens, batch_first=True, padding_value=0.0)
    batch_feats = pad_sequence(uncertainty_feats, batch_first=True, padding_value=0.0)
    
    text_lengths = torch.tensor([t.size(0) for t in text_hiddens], dtype=torch.long)
    max_text_len = batch_text.size(1)
    text_mask = torch.arange(max_text_len).expand(len(batch), max_text_len) < text_lengths.unsqueeze(1)

    batch_labels = pad_sequence(labels, batch_first=True, padding_value=pad_token_id)
    batch_label_lengths = torch.tensor(label_lengths, dtype=torch.long)
    batch_wers = torch.tensor(real_wers, dtype=torch.float)

    return (
        batch_audio,
        batch_text,
        batch_feats,
        text_mask,
        batch_labels,
        batch_label_lengths,
        batch_wers,
        speaker_ids,
        asr_texts
    )

def collate_fn_cnoisy_transformer(batch):
    vocab = ['<start>', 'ins', 'sub', 'del', 'match', '<end>']
    char2idx = {c: i for i, c in enumerate(vocab)}
    pad_token_id = -1
    model_id = "openai/whisper-large-v3"
    max_u = MAX_U

    audio_tensors = []
    audio_lengths = []
    asr_texts = []
    labels = []
    label_lengths = []
    real_wers = []
    speaker_ids = []

    for item in batch:
        wav = get_cnoisy_audio(item)
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        sr = 16000
        if sr != 16000:
            wav = torchaudio.functional.resample(wav, sr, 16000)
        audio_lengths.append(len(wav))
        wav = whisper.pad_or_trim(wav)
        audio_tensors.append(torch.tensor(wav, dtype=torch.float32))

        asr_text = item.get(f'asr_cnoisy_text_{model_id}', '')
        if asr_text is None or not isinstance(asr_text, str):
            asr_text = ""

        asr_text = asr_text.strip()
        if not asr_text:
            asr_text = "empty" 
        asr_texts.append(asr_text)

        original_seq = ast.literal_eval(item[f'asr_cnoisy_fg_multiclass_lev_{model_id}'])
        core_indices = [char2idx.get(c, char2idx['match']) for c in original_seq if c in ['ins','sub','del','match']]
        seq_indices = [char2idx['<start>']] + core_indices + [char2idx['<end>']]
        if len(seq_indices) > max_u:
            seq_indices = seq_indices[:max_u]
        labels.append(torch.tensor(seq_indices, dtype=torch.long))
        label_lengths.append(len(seq_indices))

        real_wers.append(max(0.0, min(1.0, item.get(f'asr_cnoisy_fg_wer_{model_id}', 0.0))))
        speaker_ids.append(item['fg_id'])

    padded_audio = torch.stack(audio_tensors)
    audio_pad_len = torch.tensor(padded_audio.size(1), dtype=torch.long)

    max_label_len = max(len(l) for l in labels) if labels else 1
    padded_labels = torch.full((len(batch), max_label_len), pad_token_id, dtype=torch.long)
    for i, l in enumerate(labels):
        L = len(l)
        padded_labels[i, :L] = l

    audio_lengths = torch.tensor(audio_lengths, dtype=torch.long)
    padded_pred_labels = padded_labels.clone()
    pred_label_lengths = torch.tensor(label_lengths, dtype=torch.long)

    return (
        padded_audio,
        padded_labels,
        padded_pred_labels,
        audio_lengths,
        audio_pad_len,
        torch.tensor(label_lengths, dtype=torch.long),
        pred_label_lengths,
        torch.tensor(real_wers, dtype=torch.float),
        speaker_ids,
        asr_texts
    )
