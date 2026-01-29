import torch
import torch.nn as nn

class WerGRU(nn.Module):
    def __init__(
        self,
        hidden_size=512,
    ):
        super().__init__()
        whisper_hidden_size = 1280
        self.audio_proj = nn.Linear(whisper_hidden_size, hidden_size)
        self.text_proj = nn.Linear(whisper_hidden_size, hidden_size)
        self.feat_dim = 13
        self.feat_proj = nn.Linear(self.feat_dim, hidden_size)
        
        self.wer_head = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1)
        )
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.audio_proj.weight)
        if self.audio_proj.bias is not None:
            nn.init.zeros_(self.audio_proj.bias)
        nn.init.xavier_uniform_(self.text_proj.weight)
        if self.text_proj.bias is not None:
            nn.init.zeros_(self.text_proj.bias)
        nn.init.xavier_uniform_(self.feat_proj.weight)
        if self.feat_proj.bias is not None:
            nn.init.zeros_(self.feat_proj.bias)

    def _masked_mean(self, x, mask):
        mask = mask.to(x.dtype).unsqueeze(-1)
        s = (x * mask).sum(dim=1)
        d = mask.sum(dim=1).clamp_min(1.0)
        return s / d

    def forward(self, audio_emb, text_emb, feats, text_mask):
        audio_states = self.audio_proj(audio_emb)
        audio_pool = audio_states.mean(dim=1)
        
        feat_states = self.feat_proj(feats)
        text_states = self.text_proj(text_emb)
        
        text_pool = self._masked_mean(text_states, text_mask)
        feat_pool = self._masked_mean(feat_states, text_mask)
        
        fused = torch.cat([audio_pool, text_pool, feat_pool], dim=-1)
        wer = torch.sigmoid(self.wer_head(fused)).squeeze(-1)
        return wer
