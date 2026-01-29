import torch
import torch.nn as nn
from transformers import HubertModel, AutoFeatureExtractor
from transformers import XLMRobertaModel, XLMRobertaTokenizer


# Adapted from the Fe-WER implementation (MultipleHiddenLayersModel)
# Source: https://github.com/chanhopark-research/WER-estimation/tree/main
class MultipleHiddenLayersModel(nn.Module):
    def __init__(self, layer_sizes, dropout=0.1):
        super().__init__()
        self.layer_sizes = list(layer_sizes)
        self.norms = nn.ModuleList()
        self.layers = nn.ModuleList()
        for i in range(len(self.layer_sizes) - 1):
            if i < len(self.layer_sizes) - 2:
                self.norms.append(nn.SyncBatchNorm(self.layer_sizes[i]))
            lin = nn.Linear(self.layer_sizes[i], self.layer_sizes[i+1])
            nn.init.xavier_normal_(lin.weight)
            self.layers.append(lin)
        self.function_for_hidden = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, u, t):
        out = torch.cat((u, t), dim=1)
        for i in range(len(self.layer_sizes) - 2):
            out = self.norms[i](out)
            out = self.layers[i](out)
            out = self.function_for_hidden(out)
            out = self.dropout(out)
        out = self.layers[-1](out)
        out = torch.sigmoid(out)
        return out


class WerGRU(nn.Module):
    def __init__(
        self,
        hubert_model="facebook/hubert-large-ll60k",
        xlm_model="xlm-roberta-large",
        layer_sizes=(2048, 600, 32, 1),
        dropout=0.1,
        freeze_encoders=True,
        sampling_rate=16000,
        **kwargs
    ):
        super().__init__()
        self.sampling_rate = sampling_rate

        self.feature_extractor = AutoFeatureExtractor.from_pretrained(hubert_model)
        self.hubert = HubertModel.from_pretrained(hubert_model)
        self.xlm_tok = XLMRobertaTokenizer.from_pretrained(xlm_model)
        self.xlm = XLMRobertaModel.from_pretrained(xlm_model)

        if freeze_encoders:
            for p in self.hubert.parameters():
                p.requires_grad = False
            for p in self.xlm.parameters():
                p.requires_grad = False
            self.hubert.eval()
            self.xlm.eval()

        self.audio_dim = self.hubert.config.hidden_size
        self.text_dim = self.xlm.config.hidden_size
        in_dim = self.audio_dim + self.text_dim

        if isinstance(layer_sizes, (list, tuple)):
            if layer_sizes != in_dim:
                layer_sizes = (in_dim,) + tuple(layer_sizes[1:])
        else:
            layer_sizes = (in_dim, 600, 32, 1)

        self.head = MultipleHiddenLayersModel(layer_sizes=list(layer_sizes), dropout=dropout)

    def _masked_mean(self, x, mask):
        mask = mask.to(x.dtype).unsqueeze(-1)
        s = (x * mask).sum(dim=1)
        d = mask.sum(dim=1).clamp_min(1.0)
        return s / d

    def _encode_audio(self, audio, audio_lengths=None):
        B = audio.size(0)
        audio_list = [audio[i, :audio_lengths[i]].detach().cpu().numpy() for i in range(B)]
        feats = self.feature_extractor(
            raw_speech=audio_list,
            sampling_rate=self.sampling_rate,
            padding=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        input_values = feats.input_values.to(audio.device)
        attn_mask_samples = feats.attention_mask.to(audio.device) if "attention_mask" in feats else None

        with torch.no_grad():
            hs = self.hubert(input_values=input_values, attention_mask=attn_mask_samples).last_hidden_state

        if attn_mask_samples is not None:
            B, S, _ = hs.size()
            L = attn_mask_samples.size(1)
            scale = float(S) / float(L)
            frame_lengths = (attn_mask_samples.sum(dim=1).float() * scale).ceil().clamp(min=1, max=S).long()
            frame_mask = torch.zeros(B, S, device=hs.device, dtype=hs.dtype)
            for i in range(B):
                frame_mask[i, : frame_lengths[i]] = 1.0
            audio_pool = self._masked_mean(hs, frame_mask > 0.5)
        else:
            audio_pool = hs.mean(dim=1)

        return audio_pool

    def _encode_text(self, hypothesis_inputs=None, raw_texts=None, device=None):
        if hypothesis_inputs is not None:
            if isinstance(hypothesis_inputs, (list, tuple)):
                input_ids, attention_mask = hypothesis_inputs
                tokens = {"input_ids": input_ids.to(device), "attention_mask": attention_mask.to(device)}
            elif isinstance(hypothesis_inputs, dict):
                tokens = {k: v.to(device) for k, v in hypothesis_inputs.items()}
            else:
                raise ValueError("hypothesis_inputs must be a (input_ids, attention_mask) tuple or a dict")
        elif raw_texts is not None:
            if isinstance(raw_texts, list):
                toks = self.xlm_tok(raw_texts, padding="longest", truncation=True, max_length=128, return_tensors="pt")
            else:
                toks = self.xlm_tok([raw_texts], padding="longest", truncation=True, max_length=128, return_tensors="pt")
            tokens = {k: v.to(device) for k, v in toks.items()}
        else:
            return None

        with torch.no_grad():
            hs = self.xlm(**tokens).last_hidden_state

        text_pool = self._masked_mean(hs, tokens["attention_mask"] > 0)
        return text_pool

    def get_features(self, audio):
        feats = self.feature_extractor(
            raw_speech=audio,
            return_tensors="pt",
            sampling_rate=self.sampling_rate
        )
        return feats.input_values

    def get_text_features(self, text):
        if isinstance(text, list):
            enc = self.xlm_tok(text, padding="longest", truncation=True, max_length=128, return_tensors="pt")
        else:
            enc = self.xlm_tok([text], padding="longest", truncation=True, max_length=128, return_tensors="pt")
        return enc

    def forward(
        self,
        audio,
        hypothesis_inputs=None,
        audio_lengths=None,
        hypothesis_lengths=None,
        raw_texts=None,
        target_edit_ops=None,
        real_wers=None
    ):
        device = audio.device
        u = self._encode_audio(audio, audio_lengths)
        t = self._encode_text(hypothesis_inputs=hypothesis_inputs, raw_texts=raw_texts, device=device)
        if t is None:
            t = torch.zeros(u.size(0), self.text_dim, device=device)
        wer = self.head(u, t).squeeze(-1)
        return wer, None, None
