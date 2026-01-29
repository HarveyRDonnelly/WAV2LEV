import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
import wandb
from collections import defaultdict
from tqdm import tqdm

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

    with torch.inference_mode():
        for _, batch in enumerate(val_loader):
            audio, _, _, audio_lengths, _, _, _, batch_real_wers, speaker_ids, asr_texts = batch
            audio = audio.to(device, non_blocking=True)
            audio_lengths = audio_lengths.to(device, non_blocking=True)
            batch_real_wers = batch_real_wers.to(device, non_blocking=True).clamp(0.0, 1.0)

            if isinstance(asr_texts, dict) and 'input_ids' in asr_texts and 'attention_mask' in asr_texts:
                hyp_input_ids = asr_texts['input_ids'].to(device, non_blocking=True)
                hyp_attention_mask = asr_texts['attention_mask'].to(device, non_blocking=True)
                hypothesis_inputs = (hyp_input_ids, hyp_attention_mask)
            else:
                hypothesis_inputs = None

            with torch.amp.autocast(device_type='cuda', enabled=True):
                pred_wer, _, _ = model(
                    audio=audio,
                    hypothesis_inputs=hypothesis_inputs,
                    audio_lengths=audio_lengths,
                    raw_texts=asr_texts if hypothesis_inputs is None else None,
                    real_wers=batch_real_wers
                )

            wers = pred_wer.detach().float().cpu().numpy().tolist()
            real = batch_real_wers.detach().float().cpu().numpy().tolist()

            predicted_wers.extend(wers)
            real_wers.extend(real)
            for pw, rw, spk in zip(wers, real, speaker_ids):
                speaker_wers[spk]["real"].append(rw)
                speaker_wers[spk]["predicted"].append(pw)

    rmse = np.sqrt(mean_squared_error(real_wers, predicted_wers)) if len(predicted_wers) > 0 else 0.0
    pearson_corr, p_value = pearsonr(real_wers, predicted_wers) if len(predicted_wers) > 1 else (0.0, 1.0)

    if run is not None:
        val_info = {
            "RMSE/Validation": rmse,
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
    test_data_rows = []
    progress_bar = tqdm(test_loader, desc=f"Test Evaluation")
    with torch.inference_mode():
        for _, batch in enumerate(progress_bar):
            audio, _, _, audio_lengths, _, _, _, batch_real_wers, speaker_ids, asr_texts = batch
            audio = audio.to(device, non_blocking=True)
            audio_lengths = audio_lengths.to(device, non_blocking=True)
            batch_real_wers = batch_real_wers.to(device, non_blocking=True).clamp(0.0, 1.0)
            if isinstance(asr_texts, dict) and 'input_ids' in asr_texts and 'attention_mask' in asr_texts:
                hyp_input_ids = asr_texts['input_ids'].to(device, non_blocking=True)
                hyp_attention_mask = asr_texts['attention_mask'].to(device, non_blocking=True)
                hypothesis_inputs = (hyp_input_ids, hyp_attention_mask)
            else:
                hypothesis_inputs = None
            with torch.amp.autocast(device_type='cuda', enabled=True):
                pred_wer, _, _ = model(
                    audio=audio,
                    hypothesis_inputs=hypothesis_inputs,
                    audio_lengths=audio_lengths,
                    raw_texts=asr_texts if hypothesis_inputs is None else None,
                    real_wers=batch_real_wers
                )
            wers = pred_wer.detach().float().cpu().numpy().tolist()
            real = batch_real_wers.detach().float().cpu().numpy().tolist()
            predicted_wers.extend(wers)
            real_wers.extend(real)
            for pw, rw, spk, asr_text in zip(wers, real, speaker_ids, asr_texts):
                speaker_wers[spk]["real"].append(rw)
                speaker_wers[spk]["predicted"].append(pw)
                test_data_rows.append({
                    "speaker_id": spk,
                    "asr_text": asr_text,
                    "predicted_wer": pw,
                    "real_wer": rw,
                    "error": abs(pw - rw),
                })
    rmse = np.sqrt(mean_squared_error(real_wers, predicted_wers)) if len(predicted_wers) > 0 else 0.0
    pearson_corr, p_value = pearsonr(real_wers, predicted_wers) if len(predicted_wers) > 1 else (0.0, 1.0)
    from scipy.stats import spearmanr
    spearman_corr, _ = spearmanr(real_wers, predicted_wers) if len(predicted_wers) > 1 else (0.0, 1.0)
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
        summary_table.add_data("Pearson Correlation", pearson_corr)
        summary_table.add_data("Spearman Correlation", spearman_corr)
        test_table = wandb.Table(columns=["speaker_id", "asr_text", "predicted_wer", "real_wer", "error"])
        for row in test_data_rows:
            test_table.add_data(
                row["speaker_id"],
                row["asr_text"],
                row["predicted_wer"],
                row["real_wer"],
                row["error"],
            )
        run.log({
            "Test_Summary_Metrics": summary_table,
            "Test_Detailed_Results": test_table,
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
        ax.set_title('Average WER per each speaker (test)')
        ax.set_xticks(x)
        ax.set_xticklabels(speakers, rotation=45, ha='right')
        ax.legend()
        if len(speakers) > 0:
            max_val = max(max(real_wers_avg), max(predicted_wers_avg))
            ax.set_ylim(0, max_val * 1.05)
        fig.tight_layout()
        run.log({"WER_Average_Per_Speaker": wandb.Image(fig)})
        plt.close(fig)
    return rmse, pearson_corr
