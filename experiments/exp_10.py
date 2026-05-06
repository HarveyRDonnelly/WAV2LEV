from utils.logging import Logger
import torch
import os
from torch.optim import AdamW
from torch.utils.data import DataLoader
from datasets import Dataset
from utils.config import Config
from utils.training_v11 import train
from utils.evaluation_v8 import test
from utils.data_loading import collate_fn_cnoisy_precomputed
from utils.decoding import init_transformer_vocab
from models.lev_transformer_whisper_v8 import LevTransformer as whisper_model
import pandas as pd
import numpy as np
import wandb
import dask.dataframe as dd
import datetime

os.environ["TOKENIZERS_PARALLELISM"] = "false"
c = Config()

def exp_10(model_name):
    log = Logger(script_name="Experiment 10: WAV2LEV Implementation")
    log.welcome()
    experiment_name = 'exp_10'
    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    models = {'whisper': whisper_model}
    cfg = c.EXP_MODEL_CONFIG[experiment_name]

    if model_name not in models:
        raise ValueError(f"Model name {model_name} not recognized. Choose from {list(models.keys())}.")

    model_class = models[model_name]
    run_name = f'{experiment_name}_{model_name}_{ts}'

    with wandb.init(
        project="wav2lev_final_eval",
        notes="Experiment 10: WAV2LEV Implementation",
        tags=["exp10", "WAV2LEV"],
        config=cfg,
        group=experiment_name,
        name=run_name
    ) as run:
        torch.random.manual_seed(8277388)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        df_train = dd.read_parquet(f"{c.CNOISY_DATASET_PATH}/train.parquet").compute()
        df_val = dd.read_parquet(f"{c.CNOISY_DATASET_PATH}/validation.parquet").compute()

        # Filter training set to examples with WER in a meaningful range
        wer_col = f"asr_cnoisy_fg_wer_{c.WHISPER_MODEL_ID}"
        low_thr, high_thr = 0.1, 0.95
        if wer_col in df_train.columns:
            df_train = df_train[(df_train[wer_col] > low_thr) & (df_train[wer_col] < high_thr)]

        train_dataset = Dataset.from_pandas(df_train.reset_index(drop=True))
        validation_dataset = Dataset.from_pandas(df_val.reset_index(drop=True))
        char2idx, idx2char, blank_token_id = init_transformer_vocab()

        train_loader = DataLoader(
            train_dataset,
            batch_size=cfg['batch_size'],
            shuffle=True,
            collate_fn=collate_fn_cnoisy_precomputed,
            num_workers=1,
            pin_memory=True,
            persistent_workers=True,
        )

        val_loader = DataLoader(
            validation_dataset,
            batch_size=cfg['batch_size'],
            shuffle=False,
            collate_fn=collate_fn_cnoisy_precomputed,
            num_workers=1,
            pin_memory=True,
            persistent_workers=True,
        )

        model = model_class(
            hidden_size=cfg['hidden_size'],
            num_decoder_layers=cfg['num_decoder_layers'],
            num_heads=cfg['num_heads'],
            dropout=cfg['dropout'],
            max_edit_length=cfg['max_edit_length'],
        ).to(device)

        optimizer = AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=cfg['learning_rate'],
            weight_decay=cfg['weight_decay'],
        )
        scaler = torch.cuda.amp.GradScaler()
        log.debug(f"Begin model training for {model_name}")
        results = train(
            model,
            train_loader,
            val_loader,
            optimizer,
            num_epochs=cfg['epochs'],
            device=device,
            idx2char=idx2char,
            scaler=scaler,
            log=log,
            run=run,
            label_smoothing=cfg['label_smoothing'],
            warmup_ratio=cfg['warmup_ratio'],
            gradient_accumulation_steps=cfg['gradient_accumulation_steps'],
            max_grad_norm=cfg['max_grad_norm'],
        )

        output_dir = f'out/{experiment_name}_results'
        os.makedirs(output_dir, exist_ok=True)
        results_path = f'{output_dir}/{model_name}_results.csv'
        summary_df = pd.DataFrame([{
            'model': model_name,
            'best_val_rmse': results['best_val_rmse'],
            'final_val_rmse': results['final_val_rmse'],
            'final_val_pearson': results['final_val_pearson'],
            'timestamp': ts,
        }])
        summary_df.to_csv(results_path, index=False)
        detailed_df = pd.DataFrame({
            'model': [model_name] * len(results['val_rmse_history']),
            'validation_step': range(len(results['val_rmse_history'])),
            'rmse': results['val_rmse_history'],
            'pearson': results['val_pearson_history'],
            'timestamp': [ts] * len(results['val_rmse_history']),
        })
        detailed_path = f'{output_dir}/{model_name}_detailed_results.csv'
        detailed_df.to_csv(detailed_path, index=False)
        run.save(results_path)
        run.save(detailed_path)
    log.complete()


def exp_10_test(model_name, weights_path):
    log = Logger(script_name="Experiment 10: Test Set Evaluation")
    log.welcome()
    experiment_name = 'exp_10'
    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    models = {'whisper': whisper_model}
    cfg = c.EXP_MODEL_CONFIG[experiment_name]

    if model_name not in models:
        raise ValueError(f"Model name {model_name} not recognized. Choose from {list(models.keys())}.")

    model_class = models[model_name]
    run_name = f'{experiment_name}_{model_name}_test_{ts}'

    with wandb.init(
        project="wav2lev_final_eval",
        notes="Experiment 10: Test Set Evaluation",
        tags=["exp10", "WAV2LEV", "test"],
        config=cfg,
        group=experiment_name,
        name=run_name
    ) as run:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        test_dataset = Dataset.from_pandas(
            dd.read_parquet(f"{c.CNOISY_DATASET_PATH}/test.parquet").compute())
        char2idx, idx2char, blank_token_id = init_transformer_vocab()
        test_loader = DataLoader(
            test_dataset,
            batch_size=cfg['batch_size'],
            shuffle=False,
            collate_fn=collate_fn_cnoisy_precomputed,
            num_workers=1,
            pin_memory=True,
            persistent_workers=True,
        )

        model = model_class(
            hidden_size=cfg['hidden_size'],
            num_decoder_layers=cfg['num_decoder_layers'],
            num_heads=cfg['num_heads'],
            dropout=cfg['dropout'],
            max_edit_length=cfg['max_edit_length'],
        ).to(device)
        model.load_state_dict(torch.load(weights_path, map_location=device))
        test(model, test_loader, device, idx2char, blank_token_id, log, run)
    log.complete()
