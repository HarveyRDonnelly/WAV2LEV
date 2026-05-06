import torch
import torch.nn.functional as F
from collections import namedtuple

Hypothesis = namedtuple('Hypothesis', ['score', 'state', 'sequence'])

def init_vocab():
    char2idx = {'ins': 0, 'sub': 1, 'del': 2, 'match': 3, '[BLANK]': 4}
    idx2char = {v: k for k, v in char2idx.items()}
    blank_token_id = 4
    return char2idx, idx2char, blank_token_id

def init_transformer_vocab():
    """Return (op2idx, idx2op, pad_token_id) for the LevTransformer edit vocabulary."""
    op2idx = {'<start>': 0, 'ins': 1, 'sub': 2, 'del': 3, 'match': 4, '<end>': 5, '<pad>': 6}
    idx2op = {v: k for k, v in op2idx.items()}
    pad_token_id = op2idx['<pad>']
    return op2idx, idx2op, pad_token_id


def calculate_complete_lev_wer(sequence):
    """Compute WER from a flat list of Levenshtein edit operations."""
    if not sequence:
        return 0.0
    total_ops = len([op for op in sequence if op in ['match', 'sub', 'ins', 'del']])
    if total_ops == 0:
        return 0.0
    error_ops = len([op for op in sequence if op in ['sub', 'ins', 'del']])
    return error_ops / total_ops


def beam_search_decode(log_probs,
                       idx2char,
                       blank_token_id,
                       beam_width=5,
                       max_sym_exp=10):
    """CTC beam search decoder (used by CTC-based baselines, not LevTransformer)."""
    B, T, V = log_probs.size()
    device = log_probs.device
    results = []

    for b in range(B):
        lp = log_probs[b] + 0.0
        BeamEntry = lambda seq, s: {"seq": seq, "score": s}
        beam = [BeamEntry((), 0.0)]
        for t in range(T):
            new_beam = {}
            frame = lp[t]
            topk = min(beam_width * 2, V)
            vals, ids = torch.topk(frame, topk, dim=-1)
            for entry in beam:
                s_blank = entry["score"] + frame[blank_token_id].item()
                key = entry["seq"]
                if key not in new_beam or s_blank > new_beam[key]["score"]:
                    new_beam[key] = BeamEntry(key, s_blank)
                for k in range(topk):
                    v = vals[k].item()
                    c = ids[k].item()
                    if c == blank_token_id:
                        continue
                    new_seq = entry["seq"] + (c,)
                    s_sym = entry["score"] + v
                    if new_seq not in new_beam or s_sym > new_beam[new_seq]["score"]:
                        new_beam[new_seq] = BeamEntry(new_seq, s_sym)
            beam = sorted(new_beam.values(), key=lambda x: x["score"], reverse=True)[:beam_width]

        best = max(beam, key=lambda x: x["score"])
        collapsed = []
        prev = None
        for t_idx in range(len(best["seq"])):
            sym = best["seq"][t_idx]
            if sym == prev:
                continue
            prev = sym
            if sym != blank_token_id:
                collapsed.append(sym)
        ops = [idx2char[i] for i in collapsed]
        results.append(ops)

    return results
