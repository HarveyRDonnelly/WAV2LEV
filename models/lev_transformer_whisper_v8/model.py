import torch
import torch.nn as nn
import torch.nn.functional as F

# 4, 4
# 4, 6
# 8, 8

class LevTransformer(nn.Module):
    # def __init__(
    #     self,
    #     hidden_size=512*2,
    #     num_decoder_layers=16,
    #     num_heads=16,
    #     max_edit_length=300,
    #     beam_width=2
    # ):
    def __init__(
        self,
        hidden_size=512*2,
        num_decoder_layers=12,
        num_heads=16,
        max_edit_length=300,
        beam_width=2
    ):
    # def __init__(
    #     self,
    #     hidden_size=512,
    #     num_decoder_layers=8,
    #     num_heads=8,
    #     max_edit_length=300,
    #     beam_width=2
    # ):
        super().__init__()
        self.edit_vocab = ['<start>', 'ins', 'sub', 'del', 'match', '<end>', '<pad>']
        self.vocab_size = len(self.edit_vocab)
        self.start_token_id = 0
        self.end_token_id = 5
        self.pad_token_id = 6
        self.max_edit_length = max_edit_length
        self.beam_width = beam_width

        whisper_hidden_size = 1280
        self.text_embed_dim = 1280
        
        self.audio_proj = nn.Linear(whisper_hidden_size, hidden_size)
        self.text_proj = nn.Linear(self.text_embed_dim, hidden_size)
        
        self.feat_dim = 13
        self.feat_proj = nn.Linear(self.feat_dim, hidden_size)
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=0.1,
            batch_first=True,
            activation='gelu'
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        
        self.edit_embedding = nn.Embedding(self.vocab_size, hidden_size)
        self.pos_embedding = nn.Embedding(self.max_edit_length, hidden_size)
        self.output_proj = nn.Linear(hidden_size, self.vocab_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        
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
        nn.init.xavier_uniform_(self.output_proj.weight)
        if self.output_proj.bias is not None:
            nn.init.zeros_(self.output_proj.bias)
        nn.init.normal_(self.edit_embedding.weight, mean=0, std=0.02)
        nn.init.normal_(self.pos_embedding.weight, mean=0, std=0.02)

    def generate_causal_mask(self, size):
        return torch.triu(torch.ones(size, size, device=self.output_proj.weight.device) * float('-inf'), diagonal=1)

    def forward_training(self, memory_states, target_sequences):
        targets = torch.clamp(target_sequences, 0, self.vocab_size - 1)
        B, T_full = targets.size()
        decoder_inputs = targets[:, :-1]
        targets_out = targets[:, 1:]

        embedded = self.edit_embedding(decoder_inputs)
        T = embedded.size(1)
        pos_idx = torch.arange(T, device=embedded.device)
        pos_idx = torch.clamp(pos_idx, max=self.max_edit_length - 1)
        positions = pos_idx.unsqueeze(0).expand(B, -1)
        embedded = self.layer_norm(embedded + self.pos_embedding(positions))

        tgt_mask = self.generate_causal_mask(T).to(memory_states.device)
        tgt_key_padding_mask = decoder_inputs.eq(self.pad_token_id)

        decoded = self.decoder(
            tgt=embedded,
            memory=memory_states,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask
        )
        logits = self.output_proj(decoded)
        return logits, targets_out

    def beam_search_decode(self, encoder_states, beam_width=None, max_len=None, min_len=2, len_alpha=0.7, return_logits=False):
        B = encoder_states.size(0)
        beam_width = beam_width or self.beam_width
        
        max_len = max_len or self.max_edit_length
        
        sequences = [[(torch.tensor([self.start_token_id], device=encoder_states.device, dtype=torch.long), 0.0)] for _ in range(B)]
        finished = [[] for _ in range(B)]

        for t in range(1, max_len + 1):
            new_sequences = []
            for b in range(B):
                candidates = []
                for seq, score in sequences[b]:
                    if seq[-1].item() == self.end_token_id and t > 1:
                        finished[b].append((seq, score))
                        continue
                    
                    if seq.size(0) >= max_len:
                        end_seq = torch.cat([seq, torch.tensor([self.end_token_id], device=seq.device, dtype=seq.dtype)], dim=0)
                        finished[b].append((end_seq, score))
                        continue
                    
                    emb = self.edit_embedding(seq.unsqueeze(0))
                    T = emb.size(1)
                    pos_idx = torch.arange(T, device=emb.device)
                    pos_idx = torch.clamp(pos_idx, max=self.max_edit_length - 1)
                    positions = pos_idx.unsqueeze(0)
                    emb = self.layer_norm(emb + self.pos_embedding(positions))
                    tgt_mask = self.generate_causal_mask(T).to(encoder_states.device)
                    
                    try:
                        dec = self.decoder(tgt=emb, memory=encoder_states[b:b+1], tgt_mask=tgt_mask)
                        logits = self.output_proj(dec[:, -1, :])
                        log_probs = F.log_softmax(logits, dim=-1).squeeze(0)
                        if t < min_len:
                            log_probs[self.end_token_id] = -1e9
                        topk_vals, topk_idx = torch.topk(log_probs, k=beam_width, dim=-1)
                        for k in range(beam_width):
                            nt = topk_idx[k].unsqueeze(0)
                            nscore = score + float(topk_vals[k].item())
                            candidates.append((torch.cat([seq, nt], dim=0), nscore))
                    except RuntimeError:
                        end_seq = torch.cat([seq, torch.tensor([self.end_token_id], device=seq.device, dtype=seq.dtype)], dim=0)
                        finished[b].append((end_seq, score))
                        continue
                
                if not candidates and finished[b]:
                    new_sequences.append(sorted(finished[b], key=lambda x: x[1], reverse=True)[:beam_width])
                    continue
                if not candidates:
                    seq = torch.tensor([self.start_token_id, self.end_token_id], device=encoder_states.device, dtype=torch.long)
                    new_sequences.append([(seq, 0.0)])
                    continue
                
                ordered = sorted(candidates, key=lambda x: x[-1], reverse=True)
                new_sequences.append(ordered[:beam_width])
            
            sequences = new_sequences

            for b in range(B):
                pruned = []
                for seq, score in sequences[b]:
                    if seq[-1].item() == self.end_token_id and seq.size(0) >= min_len:
                        L = seq.size(0)
                        norm_score = score / ((5 + L) ** len_alpha / (5 + 1) ** len_alpha)
                        finished[b].append((seq, norm_score))
                    else:
                        pruned.append((seq, score))
                sequences[b] = pruned if pruned else sequences[b]
            all_done = all(len(sequences[b]) == 0 for b in range(B))
            if all_done:
                break

        results = []
        for b in range(B):
            pool = finished[b] if finished[b] else (sequences[b] if sequences[b] else [])
            if not pool:
                seq = torch.tensor([self.start_token_id, self.end_token_id], device=encoder_states.device, dtype=torch.long)
                results.append(seq.unsqueeze(0))
            else:
                best_seq, _ = max(pool, key=lambda x: x[1])
                results.append(best_seq.unsqueeze(0))

        maxT = max(seq.size(1) for seq in results)
        out_ids = torch.full((B, maxT), fill_value=self.end_token_id, dtype=torch.long, device=encoder_states.device)
        for b, seq in enumerate(results):
            L = seq.size(1)
            out_ids[b, :L] = seq

        if not return_logits:
            return out_ids, None

        all_logits = []
        for b in range(B):
            seq = out_ids[b:b+1]
            emb = self.edit_embedding(seq)
            T = emb.size(1)
            pos_idx = torch.arange(T, device=emb.device)
            pos_idx = torch.clamp(pos_idx, max=self.max_edit_length - 1)
            positions = pos_idx.unsqueeze(0)
            emb = self.layer_norm(emb + self.pos_embedding(positions))
            tgt_mask = self.generate_causal_mask(T).to(encoder_states.device)
            dec = self.decoder(tgt=emb, memory=encoder_states[b:b+1], tgt_mask=tgt_mask)
            logits = self.output_proj(dec)
            all_logits.append(logits)
        out_logits = torch.cat(all_logits, dim=0)
        return out_ids, out_logits

    def forward(self, audio_emb, text_emb, feats, target_edit_ops=None):
        audio_states = self.audio_proj(audio_emb)
        feat_states = self.feat_proj(feats)
        text_states = self.text_proj(text_emb)
        
        memory_states = torch.cat([audio_states, text_states, feat_states], dim=1)

        if self.training and target_edit_ops is not None:
            logits, targets_out = self.forward_training(memory_states, target_edit_ops)
            return logits, targets_out
        else:
            pred_ids, true_logits = self.beam_search_decode(
                memory_states, beam_width=self.beam_width, max_len=self.max_edit_length, return_logits=True
            )
            if true_logits is not None:
                return true_logits, None
            B, T = pred_ids.size(0), pred_ids.size(1)
            logits = torch.full((B, T, self.vocab_size), -1e4, device=memory_states.device)
            logits.scatter_(2, pred_ids.unsqueeze(-1), 0.0)
            return logits, None
