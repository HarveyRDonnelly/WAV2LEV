import torch
from utils.evaluation_v4 import validate
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
import math
import gc
from utils.config import Config
import wandb

c = Config()

def train(model,
          train_loader,
          val_loader,
          optimizer,
          num_epochs,
          device,
          idx2char,
          scaler,
          log,
          run,
          num_val_per_epoch=1,
          gradient_accumulation_steps=2,
          max_grad_norm=1.0):

    start_epoch = 0
    global_step = 0
    best_val_rmse = float('inf')

    steps_per_epoch = len(train_loader)
    total_updates = steps_per_epoch * num_epochs // max(1, gradient_accumulation_steps)
    warmup_steps = int(0.01 * max(1, total_updates))
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=max(1, total_updates)
    )

    criterion = torch.nn.MSELoss(reduction='mean')

    val_interval = max(1, steps_per_epoch // max(1, num_val_per_epoch))

    for epoch in range(start_epoch, num_epochs):
        model.train()
        train_loss_accum = 0.0
        progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")

        for batch_idx, batch in enumerate(progress):
            audio, _, _, audio_lengths, _, _, _, real_wers, _, asr_texts = batch
            audio = audio.to(device, non_blocking=True)
            audio_lengths = audio_lengths.to(device, non_blocking=True)
            real_wers = real_wers.to(device, non_blocking=True).clamp(0.0, 1.0)

            if isinstance(asr_texts, dict) and 'input_ids' in asr_texts and 'attention_mask' in asr_texts:
                hyp_input_ids = asr_texts['input_ids'].to(device, non_blocking=True)
                hyp_attention_mask = asr_texts['attention_mask'].to(device, non_blocking=True)
                hypothesis_inputs = (hyp_input_ids, hyp_attention_mask)
            else:
                hypothesis_inputs = None

            with torch.amp.autocast('cuda'):
                pred_wer, _, _ = model(
                    audio=audio,
                    hypothesis_inputs=hypothesis_inputs,
                    audio_lengths=audio_lengths,
                    real_wers=real_wers,
                    raw_texts=asr_texts if hypothesis_inputs is None else None
                )
                loss = criterion(pred_wer, real_wers) / max(1, gradient_accumulation_steps)

            scaler.scale(loss).backward()

            do_update = ((batch_idx + 1) % gradient_accumulation_steps == 0) or ((batch_idx + 1) == len(train_loader))
            if do_update:
                scaler.unscale_(optimizer)
                if max_grad_norm and max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_step += 1

                run.log({
                    "Loss/Training": loss.item() * max(1, gradient_accumulation_steps),
                    "LR/Training": float(scheduler.get_last_lr()[0]),
                    "Epoch": epoch
                }, global_step)

            train_loss_accum += loss.item() * max(1, gradient_accumulation_steps)
            progress.set_postfix({
                "loss": f"{loss.item() * max(1, gradient_accumulation_steps):.4f}",
                "root_loss": f"{math.sqrt(max(loss.item() * max(1, gradient_accumulation_steps), 0.0)):.4f}",
                "lr": f"{scheduler.get_last_lr()[0]:.6f}"
            })

            del loss
            torch.cuda.empty_cache()

            # if num_val_per_epoch > 0 and (batch_idx + 1) % val_interval == 0 and (batch_idx + 1) < len(train_loader):
            #     with torch.inference_mode():
            #         with torch.amp.autocast(device_type='cuda', enabled=True):
            #             val_rmse, val_pearson = validate(model, val_loader, device, idx2char, -1, log, run, global_step)
            #     if val_rmse < best_val_rmse:
            #         best_val_rmse = val_rmse
            #         torch.save(model.state_dict(), f"{c.WEIGHTS_PATH}/{run.name}_best.pt")
            #     model.train()

        avg_train_loss = train_loss_accum / max(1, len(train_loader))
        progress.clear()
        progress.write(f"Epoch {epoch+1}/{num_epochs} completed:")
        progress.write(f"  Train Loss: {avg_train_loss:.4f}")
        progress.display()
        torch.cuda.empty_cache()

        if num_val_per_epoch > 0:
            with torch.inference_mode():
                with torch.amp.autocast(device_type='cuda', enabled=True):
                    val_rmse, val_pearson = validate(model, val_loader, device, idx2char, -1, log, run, global_step)
            if val_rmse < best_val_rmse:
                best_val_rmse = val_rmse
                torch.save(model.state_dict(), f"{c.WEIGHTS_PATH}/{run.name}_best.pt")

        torch.save(model.state_dict(), f"{c.WEIGHTS_PATH}/{run.name}_recent.pt")
        gc.collect()
        torch.cuda.empty_cache()

    return {
        'val_rmse_history': [],
        'val_pearson_history': [],
        'best_val_rmse': best_val_rmse,
        'final_val_rmse': val_rmse if 'val_rmse' in locals() else best_val_rmse,
        'final_val_pearson': val_pearson if 'val_pearson' in locals() else 0.0
    }
