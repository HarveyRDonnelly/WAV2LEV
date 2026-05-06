import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from datasets import Dataset
import dask.dataframe as dd
from tqdm import tqdm
from utils.config import Config
from utils.data_loading import collate_fn_cnoisy_transformer
from nemo_text_processing.text_normalization import Normalizer as NeMoNormalizer
from utils.text_normalisation import CustomNormaliser

c = Config()


def compute_uncertainty_features(logits, input_ids):
    """Compute 13 per-token uncertainty features from Whisper decoder output logits.

    Features (in order):
      p1           — top-1 token probability
      p2           — top-2 token probability
      margin       — p1 - p2
      ratio        — p2 / (p1 + ε)
      norm_entropy — Shannon entropy normalised by log(vocab_size)
      gini         — Gini impurity: 1 - Σp²
      hhi          — Herfindahl–Hirschman index: Σp² (concentration measure)
      top3         — cumulative probability mass of top-3 tokens
      top5         — cumulative probability mass of top-5 tokens
      top10        — cumulative probability mass of top-10 tokens
      neff         — normalised effective vocabulary: exp(H) / V
      tgt_prob     — probability assigned to the hypothesis token
      nll          — negative log-likelihood of the hypothesis token
    """
    with torch.no_grad():
        probs = F.softmax(logits, dim=-1)
        V = probs.size(-1)
        if input_ids.size(1) != probs.size(1):
            T = min(input_ids.size(1), probs.size(1))
            input_ids = input_ids[:, :T]
            probs = probs[:, :T, :]

        topk_vals, _ = torch.topk(probs, k=min(10, V), dim=-1)
        p1 = topk_vals[..., 0]
        p2 = topk_vals[..., 1] if topk_vals.size(-1) > 1 else torch.zeros_like(p1)

        margin = p1 - p2
        ratio = p2 / (p1 + 1e-8)

        entropy = -(probs * probs.clamp_min(1e-12).log()).sum(dim=-1)
        norm_entropy = entropy / float(torch.log(torch.tensor(V, device=logits.device)))

        sq_sum = (probs * probs).sum(dim=-1)
        gini = 1.0 - sq_sum
        hhi = sq_sum

        top3 = topk_vals[..., :min(3, V)].sum(dim=-1)
        top5 = topk_vals[..., :min(5, V)].sum(dim=-1)
        top10 = topk_vals[..., :min(10, V)].sum(dim=-1)

        neff = torch.exp(entropy) / float(V)

        tgt_probs = probs.gather(dim=-1, index=input_ids.unsqueeze(-1)).squeeze(-1)
        nll_tgt = -(tgt_probs.clamp_min(1e-12).log())

        feats = torch.stack(
            [p1, p2, margin, ratio, norm_entropy, gini, hhi, top3, top5, top10, neff, tgt_probs, nll_tgt],
            dim=-1,
        )
    return feats


def _resume_state(save_dir):
    """Return the set of already-processed fg_ids, deleting the last written file.

    The last file is removed because it may have been partially written if the
    previous run was interrupted mid-save (before os.replace completed).
    """
    os.makedirs(save_dir, exist_ok=True)

    for fn in os.listdir(save_dir):
        if fn.endswith(".tmp"):
            try:
                os.remove(os.path.join(save_dir, fn))
            except OSError:
                pass

    pt_files = [fn for fn in os.listdir(save_dir) if fn.endswith(".pt")]
    if not pt_files:
        return set()

    last_fn = max(pt_files, key=lambda f: os.path.getmtime(os.path.join(save_dir, f)))
    last_path = os.path.join(save_dir, last_fn)
    try:
        os.remove(last_path)
    except OSError:
        pass

    done = {os.path.splitext(fn)[0] for fn in pt_files}
    done.discard(os.path.splitext(last_fn)[0])
    return done


def precompute_embeddings():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_name = c.WHISPER_MODEL_ID

    processor = WhisperProcessor.from_pretrained(model_name)
    model = WhisperForConditionalGeneration.from_pretrained(model_name).to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    nemo_norm = NeMoNormalizer(lang="en", input_case="lower_cased", deterministic=True, cache_dir=".tn_cache")
    normalizer = CustomNormaliser(nemo_norm)

    splits = ["train", "validation", "test"]
    save_dir = c.EMBEDDINGS_PRECOMP_PATH
    done_ids = _resume_state(save_dir)

    for split in splits:
        print(f"Processing {split} split...")
        df = dd.read_parquet(f"{c.CNOISY_DATASET_PATH}/{split}.parquet").compute()
        dataset = Dataset.from_pandas(df)

        loader = DataLoader(
            dataset,
            batch_size=8,
            shuffle=False,
            collate_fn=collate_fn_cnoisy_transformer,
            num_workers=4,
            pin_memory=True,
        )

        for batch in tqdm(loader):
            (padded_audio, padded_labels, _, audio_lengths, _, _, _, _, speaker_ids, asr_texts) = batch

            speaker_ids_str = [str(x) for x in speaker_ids]
            idxs = [j for j, sid in enumerate(speaker_ids_str) if sid not in done_ids]
            if not idxs:
                continue

            idx_t = torch.tensor(idxs, dtype=torch.long)

            padded_audio = padded_audio.index_select(0, idx_t)
            asr_texts = [asr_texts[j] for j in idxs]
            speaker_ids_str = [speaker_ids_str[j] for j in idxs]

            normalized_texts = [normalizer(text) for text in asr_texts]

            input_features = processor.feature_extractor(
                padded_audio.numpy(),
                sampling_rate=16000,
                return_tensors="pt",
            ).input_features.to(device)

            text_inputs = processor.tokenizer(
                normalized_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=448,
            )
            input_ids = text_inputs.input_ids.to(device)
            attention_mask = text_inputs.attention_mask.to(device)

            with torch.no_grad():
                encoder_outputs = model.model.encoder(
                    input_features=input_features,
                    attention_mask=None,
                )
                audio_hidden = encoder_outputs.last_hidden_state

                outputs = model(
                    encoder_outputs=encoder_outputs,
                    decoder_input_ids=input_ids,
                    decoder_attention_mask=attention_mask,
                    output_attentions=False,
                    return_dict=True,
                )
                logits = outputs.logits
                token_emb = model.model.decoder.embed_tokens(input_ids)
                feats = compute_uncertainty_features(logits, input_ids)

            B = input_features.size(0)
            for i in range(B):
                fg_id = speaker_ids_str[i]
                cur_len = int(attention_mask[i].sum().item())

                save_path = os.path.join(save_dir, f"{fg_id}.pt")
                tmp_path = save_path + ".tmp"

                torch.save(
                    {
                        "audio_hidden": audio_hidden[i].cpu().clone(),
                        "text_hidden": token_emb[i, :cur_len].cpu().clone(),
                        "uncertainty_feats": feats[i, :cur_len].cpu().clone(),
                        "input_ids": input_ids[i, :cur_len].cpu().clone(),
                        "asr_text": normalized_texts[i],
                    },
                    tmp_path,
                )
                os.replace(tmp_path, save_path)
                done_ids.add(fg_id)


if __name__ == "__main__":
    precompute_embeddings()
