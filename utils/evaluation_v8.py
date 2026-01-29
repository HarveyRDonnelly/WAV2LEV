import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error
import wandb
from collections import defaultdict
from tqdm import tqdm

def _levenshtein_distance(a, b):
    na, nb = len(a), len(b)
    if na == 0:
        return nb
    if nb == 0:
        return na
    dp = list(range(nb + 1))
    for i in range(1, na + 1):
        prev = dp[0]
        dp[0] = i
        ai = a[i - 1]
        for j in range(1, nb + 1):
            tmp = dp[j]
            cost = 0 if ai == b[j - 1] else 1
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + cost)
            prev = tmp
    return dp[nb]

def _compute_pred_wer(pred_ops):
    n_ins = sum(1 for op in pred_ops if op == 'ins')
    n_sub = sum(1 for op in pred_ops if op == 'sub')
    n_del = sum(1 for op in pred_ops if op == 'del')
    n_match = sum(1 for op in pred_ops if op == 'match')
    ref_len = n_match + n_sub + n_del
    err = n_ins + n_sub + n_del
    if ref_len <= 0:
        return 0.0
    wer = float(err) / float(ref_len)
    wer = min(wer, 1.0)
    return wer

def _greedy_decode_ops(logits, idx2char):
    probs = torch.log_softmax(logits.float(), dim=-1).exp()
    ids = probs.argmax(dim=-1).detach().cpu().tolist()
    seqs = []
    for row in ids:
        ops = []
        for t in row:
            sym = idx2char.get(t, None)
            if sym is None:
                continue
            if sym == '<start>':
                continue
            if sym == '<end>':
                break
            ops.append(sym)
        seqs.append(ops)
    return seqs

def validate(model,
             val_loader,
             device,
             idx2char,
             blank_token_id,
             log,
             run=None,
             step=0):
    log.debug(f'Validation at step {step}.')
    model.eval()

    predicted_wers = []
    real_wers = []
    speaker_wers = defaultdict(lambda: {"real": [], "predicted": []})
    correct_predictions = 0
    total_predictions = 0
    pred_seq_lengths = []
    target_seq_lengths = []

    with torch.inference_mode():
        for _, batch in enumerate(val_loader):
            audio_emb, text_emb, feats, _, target_edit_ops, target_lengths, batch_real_wers, speaker_ids, asr_texts = batch

            audio_emb = audio_emb.to(device)
            text_emb = text_emb.to(device)
            feats = feats.to(device)
            target_edit_ops = target_edit_ops.to(device)
            target_lengths = target_lengths.to(device)

            model.beam_width = 1
            logits, _ = model(
                audio_emb=audio_emb,
                text_emb=text_emb,
                feats=feats,
                target_edit_ops=None
            )

            decoded_preds = _greedy_decode_ops(logits, idx2char)

            for i, (pred_ops, real_wer, speaker_id) in enumerate(zip(decoded_preds, batch_real_wers, speaker_ids)):
                pred_wer = _compute_pred_wer(pred_ops)
                speaker_wers[speaker_id]["real"].append(real_wer.item())
                speaker_wers[speaker_id]["predicted"].append(pred_wer)
                predicted_wers.append(pred_wer)
                real_wers.append(real_wer.item())

                tgt_ops = []
                for idx in target_edit_ops[i, :target_lengths[i]]:
                    if idx.item() in idx2char:
                        tgt_ops.append(idx2char[idx.item()])

                pred_len = len([op for op in pred_ops if op in ['match', 'sub', 'ins', 'del']])
                tgt_len = len([op for op in tgt_ops if op in ['match', 'sub', 'ins', 'del']])
                pred_seq_lengths.append(pred_len)
                target_seq_lengths.append(tgt_len)

                min_len = min(len(pred_ops), len(tgt_ops))
                correct_predictions += sum(1 for j in range(min_len) if pred_ops[j] == tgt_ops[j])
                total_predictions += min_len

    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    rmse = np.sqrt(mean_squared_error(real_wers, predicted_wers)) if len(predicted_wers) > 0 else 0.0
    pearson_corr, p_value = pearsonr(real_wers, predicted_wers) if len(predicted_wers) > 1 else (0.0, 1.0)

    if run is not None:
        val_info = {
            "RMSE/Validation": rmse,
            "Accuracy/Validation": accuracy,
            "Pearson/Validation": pearson_corr,
        }
        run.log(val_info, step)

        fig = plt.figure(figsize=(10, 10))
        if len(real_wers) > 0 and len(predicted_wers) > 0:
            plt.scatter(real_wers, predicted_wers, alpha=0.5)
            plt.xlabel('Real WER')
            plt.ylabel('Predicted WER')
            plt.title(f'Predicted vs Real WER (step {step})')
            min_val = min(min(real_wers), min(predicted_wers))
            max_val = max(max(real_wers), max(predicted_wers))
            plt.plot([min_val, max_val], [min_val, max_val], 'r--')
            run.log({"WER_Correlation": wandb.Image(fig)}, step=step)
        plt.close(fig)

        fig = plt.figure(figsize=(10, 6))
        if len(pred_seq_lengths) > 0 and len(target_seq_lengths) > 0:
            bins = max(10, int(np.sqrt(len(pred_seq_lengths))))
            plt.hist(pred_seq_lengths, bins=bins, alpha=0.6, label='predicted', color='#ff7f0e')
            plt.hist(target_seq_lengths, bins=bins, alpha=0.6, label='target', color='#1f77b4')
            plt.xlabel('Sequence length (ops)')
            plt.ylabel('Count')
            plt.title(f'Predicted vs Target Sequence Lengths (step {step})')
            plt.legend()
            run.log({"SeqLen_Hist_Validation": wandb.Image(fig)}, step=step)
        plt.close(fig)

        speakers = list(speaker_wers.keys())
        real_wers_avg = [np.mean(speaker_wers[s]["real"]) for s in speakers] if len(speakers) > 0 else []
        predicted_wers_avg = [np.mean(speaker_wers[s]["predicted"]) for s in speakers] if len(speakers) > 0 else []

        x = np.arange(len(speakers))
        width = 0.35
        fig, ax = plt.subplots(figsize=(15, 10))
        if len(speakers) > 0:
            ax.bar(x - width/2, real_wers_avg, width, label='target', color='#1f77b4')
            ax.bar(x + width/2, predicted_wers_avg, width, label='estimate', color='#ff7f0e')
        ax.set_ylabel('WER')
        ax.set_title(f'Average WER per each speaker (step {step})')
        ax.set_xticks(x)
        ax.set_xticklabels(speakers, rotation=45, ha='right')
        ax.legend()
        if len(speakers) > 0:
            max_val = max(max(real_wers_avg), max(predicted_wers_avg))
            ax.set_ylim(0, max_val * 1.05)
        fig.tight_layout()
        run.log({"WER_Average_Per_Speaker": wandb.Image(fig)}, step=step)
        plt.close(fig)

    return rmse, pearson_corr

def test(model,
         test_loader,
         device,
         idx2char,
         blank_token_id,
         log,
         run=None):
    
    model.eval()
    predicted_wers = []
    real_wers = []
    speaker_wers = defaultdict(lambda: {"real": [], "predicted": []})
    correct_predictions = 0
    total_predictions = 0
    test_data_rows = []
    pred_ops_counts = defaultdict(int)
    target_ops_counts = defaultdict(int)
    pred_seq_lengths = []
    target_seq_lengths = []
    total_token_edits = 0
    total_target_tokens = 0
    lev_length_pred = []
    lev_length_tgt = []
    groundtruth_length_pred = []
    groundtruth_length_tgt = []
    
    progress_bar = tqdm(test_loader, desc=f"Test Evaluation")
    
    with torch.inference_mode():
        for _, batch in enumerate(progress_bar):
            audio_emb, text_emb, feats, _, target_edit_ops, target_lengths, batch_real_wers, speaker_ids, asr_texts = batch
            
            audio_emb = audio_emb.to(device)
            text_emb = text_emb.to(device)
            feats = feats.to(device)
            target_edit_ops = target_edit_ops.to(device)
            target_lengths = target_lengths.to(device)
            
            model.beam_width = 1
            logits, _ = model(
                audio_emb=audio_emb,
                text_emb=text_emb,
                feats=feats,
                target_edit_ops=None
            )
            
            decoded_preds = _greedy_decode_ops(logits, idx2char)
            asr_text_list = list(asr_texts)
            
            for i, (pred_ops, real_wer, speaker_id, asr_text) in enumerate(zip(decoded_preds, batch_real_wers, speaker_ids, asr_text_list)):
                pred_wer = _compute_pred_wer(pred_ops)
                speaker_wers[speaker_id]["real"].append(real_wer.item())
                speaker_wers[speaker_id]["predicted"].append(pred_wer)
                predicted_wers.append(pred_wer)
                real_wers.append(real_wer.item())
                
                for op in pred_ops:
                    pred_ops_counts[op] += 1
                
                target_seq = []
                for idx in target_edit_ops[i, :target_lengths[i]]:
                    if idx.item() in idx2char:
                        target_seq.append(idx2char[idx.item()])
                
                for op in target_seq:
                    target_ops_counts[op] += 1
                
                pred_len = len([op for op in pred_ops if op in ['match', 'sub', 'ins', 'del']])
                tgt_len = len([op for op in target_seq if op in ['match', 'sub', 'ins', 'del']])
                pred_seq_lengths.append(pred_len)
                target_seq_lengths.append(tgt_len)
                
                test_data_rows.append({
                    "speaker_id": speaker_id,
                    "asr_text": asr_text,
                    "predicted_wer": pred_wer,
                    "real_wer": real_wer.item(),
                    "error": abs(pred_wer - real_wer.item()),
                    "pred_lev_seq": ' '.join(pred_ops),
                    "target_lev_seq": ' '.join(target_seq)
                })
                
                min_len = min(len(pred_ops), len(target_seq))
                correct_predictions += sum(1 for j in range(min_len) if pred_ops[j] == target_seq[j])
                total_predictions += min_len
                
                total_token_edits += _levenshtein_distance(target_seq, pred_ops)
                total_target_tokens += len(target_seq)
                
                n_match_p = sum(1 for op in pred_ops if op == 'match')
                n_sub_p = sum(1 for op in pred_ops if op == 'sub')
                n_del_p = sum(1 for op in pred_ops if op == 'del')
                
                n_match_t = sum(1 for op in target_seq if op == 'match')
                n_sub_t = sum(1 for op in target_seq if op == 'sub')
                n_del_t = sum(1 for op in target_seq if op == 'del')
                
                lev_length_pred.append(pred_len)
                lev_length_tgt.append(tgt_len)
                
                groundtruth_length_pred.append(n_match_p + n_sub_p + n_del_p)
                groundtruth_length_tgt.append(n_match_t + n_sub_t + n_del_t)

    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    rmse = np.sqrt(mean_squared_error(real_wers, predicted_wers)) if len(predicted_wers) > 0 else 0.0
    pearson_corr, p_value = pearsonr(real_wers, predicted_wers) if len(real_wers) > 1 else (0.0, 1.0)
    spearman_corr, _ = spearmanr(real_wers, predicted_wers) if len(real_wers) > 1 else (0.0, 1.0)
    
    token_error_rate = (float(total_token_edits) / float(max(1, total_target_tokens))) if total_target_tokens > 0 else 0.0
    
    lev_length_rmse = np.sqrt(mean_squared_error(lev_length_tgt, lev_length_pred)) if len(lev_length_pred) > 0 else 0.0
    lev_length_pcc, _ = pearsonr(lev_length_tgt, lev_length_pred) if len(lev_length_pred) > 1 else (0.0, 1.0)
    
    groundtruth_length_rmse = np.sqrt(mean_squared_error(groundtruth_length_tgt, groundtruth_length_pred)) if len(groundtruth_length_pred) > 0 else 0.0
    groundtruth_length_pcc, _ = pearsonr(groundtruth_length_tgt, groundtruth_length_pred) if len(groundtruth_length_pred) > 1 else (0.0, 1.0)
    
    def _nrmse(y_true, y_pred, scale='mean'):
        if len(y_true) == 0:
            return 0.0, 0.0
        y_true_np = np.array(y_true, dtype=float)
        y_pred_np = np.array(y_pred, dtype=float)
        rmse_val = np.sqrt(np.mean((y_pred_np - y_true_np) ** 2))
        if scale == 'mean':
            denom = np.maximum(np.mean(np.abs(y_true_np)), 1e-8)
        elif scale == 'std':
            denom = np.maximum(np.std(y_true_np), 1e-8)
        else:
            denom = np.maximum(np.max(y_true_np) - np.min(y_true_np), 1e-8)
        nrmse_val = rmse_val / denom
        abs_err = np.abs(y_pred_np - y_true_np)
        nrmse_itemwise = abs_err / np.maximum(denom, 1e-8)
        return float(nrmse_val), float(np.std(nrmse_itemwise))
    
    lev_length_nrmse_mean, lev_length_nrmse_std = _nrmse(lev_length_tgt, lev_length_pred, scale='mean')
    groundtruth_length_nrmse_mean, groundtruth_length_nrmse_std = _nrmse(groundtruth_length_tgt, groundtruth_length_pred, scale='mean')
    
    def _hist_kl(p_vals, q_vals, bins=20, rng=None, eps=1e-12):
        if len(p_vals) == 0 or len(q_vals) == 0:
            return 0.0
        if rng is None:
            both = np.array(list(p_vals) + list(q_vals), dtype=float)
            rng = (float(both.min()), float(both.max()))
        hist_p, edges = np.histogram(p_vals, bins=bins, range=rng, density=True)
        hist_q, _ = np.histogram(q_vals, bins=edges, density=True)
        p = hist_p + eps
        q = hist_q + eps
        p /= p.sum()
        q /= q.sum()
        return float(np.sum(p * np.log(p / q)))
    
    lev_length_kl = _hist_kl(lev_length_tgt, lev_length_pred, bins=max(10, int(np.sqrt(max(1, len(lev_length_tgt) + len(lev_length_pred))))))
    groundtruth_length_kl = _hist_kl(groundtruth_length_tgt, groundtruth_length_pred, bins=max(10, int(np.sqrt(max(1, len(groundtruth_length_tgt) + len(groundtruth_length_pred))))))
    
    dataset_size = len(predicted_wers)
    mean_pred_wer = np.mean(predicted_wers) if len(predicted_wers) > 0 else 0.0
    std_pred_wer = np.std(predicted_wers) if len(predicted_wers) > 0 else 0.0
    mean_real_wer = np.mean(real_wers) if len(real_wers) > 0 else 0.0
    std_real_wer = np.std(real_wers) if len(real_wers) > 0 else 0.0
    
    if run is not None:
        summary_table = wandb.Table(columns=["Metric", "Value"])
        summary_table.add_data("Dataset Size", dataset_size)
        summary_table.add_data("Mean Predicted WER", mean_pred_wer)
        summary_table.add_data("Std Predicted WER", std_pred_wer)
        summary_table.add_data("Mean Real WER", mean_real_wer)
        summary_table.add_data("Std Real WER", std_real_wer)
        summary_table.add_data("RMSE", rmse)
        summary_table.add_data("Accuracy", accuracy)
        summary_table.add_data("Pearson Correlation", pearson_corr)
        summary_table.add_data("Spearman Correlation", spearman_corr)
        summary_table.add_data("Token Error Rate", token_error_rate)
        summary_table.add_data("lev_length_RMSE", lev_length_rmse)
        summary_table.add_data("lev_length_Pearson", lev_length_pcc)
        summary_table.add_data("groundtruth_length_RMSE", groundtruth_length_rmse)
        summary_table.add_data("groundtruth_length_Pearson", groundtruth_length_pcc)
        summary_table.add_data("lev_length_NRMSE_mean", lev_length_nrmse_mean)
        summary_table.add_data("lev_length_NRMSE_std", lev_length_nrmse_std)
        summary_table.add_data("groundtruth_length_NRMSE_mean", groundtruth_length_nrmse_mean)
        summary_table.add_data("groundtruth_length_NRMSE_std", groundtruth_length_nrmse_std)
        summary_table.add_data("lev_length_KL", lev_length_kl)
        summary_table.add_data("groundtruth_length_KL", groundtruth_length_kl)
        
        test_table = wandb.Table(columns=["speaker_id", "asr_text", "predicted_wer", "real_wer", "error", "pred_lev_seq", "target_lev_seq"])
        for row in test_data_rows:
            test_table.add_data(
                row["speaker_id"],
                row["asr_text"],
                row["predicted_wer"],
                row["real_wer"],
                row["error"],
                row["pred_lev_seq"],
                row["target_lev_seq"]
            )
            
        run.log({
            "Test_Summary_Metrics": summary_table,
            "Test_Detailed_Results": test_table,
            "Predicted_Operation_Counts": wandb.Histogram(list(pred_ops_counts.values())),
            "Target_Operation_Counts": wandb.Histogram(list(target_ops_counts.values())),
            "Predicted_Sequence_lengths": wandb.Histogram(pred_seq_lengths),
            "Target_Sequence_lengths": wandb.Histogram(target_seq_lengths)
        })
        
        fig = plt.figure(figsize=(10, 10))
        if len(real_wers) > 0 and len(predicted_wers) > 0:
            x = np.array(real_wers)
            y = np.array(predicted_wers)
            plt.scatter(x, y, alpha=0.5)
            m, b = np.polyfit(x, y, 1)
            x_line = np.linspace(min(x), max(x), 100)
            plt.plot(x_line, m * x_line + b, 'g-', linewidth=2)
            plt.xlabel('Target WER', fontsize=16)
            plt.ylabel('Predicted WER', fontsize=16)
            run.log({"WER_Correlation": wandb.Image(fig)})
        plt.close(fig)
        
        fig = plt.figure(figsize=(10, 6))
        if len(pred_seq_lengths) > 0 and len(target_seq_lengths) > 0:
            all_lengths = np.array(target_seq_lengths + pred_seq_lengths, dtype=float)
            bins = max(10, int(np.sqrt(len(all_lengths))))
            rng = (float(all_lengths.min()), float(all_lengths.max()))
            plt.hist(target_seq_lengths, bins=bins, range=rng, alpha=0.6, label='Target Lev. Sequences', color='#1f77b4', density=True)
            plt.hist(pred_seq_lengths, bins=bins, range=rng, alpha=0.6, label='Predicted Lev. Sequences', color='#ff7f0e', density=True)
            plt.xlabel('Sequence Length', fontsize=16)
            plt.ylabel('Frequency', fontsize=16)
            plt.legend()
            run.log({"SeqLen_Distribution_Test": wandb.Image(fig)})
        plt.close(fig)
        
        fig = plt.figure(figsize=(10, 6))
        if len(real_wers) > 0:
            bins = 20
            data_percent = [w * 100.0 for w in real_wers]
            plt.hist(data_percent, bins=bins, color="#d0479a", edgecolor='black', linewidth=1.2, alpha=0.8, density=True)
            plt.xlabel('Target WER', fontsize=16)
            plt.ylabel('Frequency', fontsize=16)
            run.log({"WER_Distribution_Test": wandb.Image(fig)})
        plt.close(fig)
        
        fig = plt.figure(figsize=(10, 6))
        if len(predicted_wers) > 0:
            bins = 20
            data_percent_pred = [w * 100.0 for w in predicted_wers]
            plt.hist(data_percent_pred, bins=bins, color="#d0479a", edgecolor='black', linewidth=1.2, alpha=0.8, density=True)
            plt.xlabel('Target WER', fontsize=16)
            plt.ylabel('Frequency', fontsize=16)
            run.log({"Predicted_WER_Distribution_Test": wandb.Image(fig)})
        plt.close(fig)
        
    return rmse, pearson_corr
