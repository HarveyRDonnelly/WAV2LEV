import argparse
from utils.logging import Logger
import torch
import os, sys
from torch.optim import AdamW
from torch.utils.data import DataLoader
from datasets import Dataset
from utils.config import Config
from utils.training_v10 import train
from utils.evaluation_v7 import test
from utils.data_loading import collate_fn_cnoisy_precomputed
from utils.decoding import init_transformer_vocab
from models.wer_gru_whisper_v4 import WerGRU as whisper_model
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import wandb
import dask.dataframe as dd
import datetime
import os
from datasets import load_from_disk

os.environ["TOKENIZERS_PARALLELISM"] = "false"
c = Config()

def exp_9(model_name):
    
    log = Logger(script_name="Experiment 9: WHISP-MLP Implementation")
    log.welcome()
    
    experiment_name = 'exp_9'
    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    
    models = {
        'whisper': whisper_model
    }
    
    if model_name not in models:
        raise ValueError(f"Model name {model_name} not recognized. Choose from {list(models.keys())}.")
    
    model_class = models[model_name]
        
    run_name = f'{experiment_name}_{model_name}_{ts}'
    
    with wandb.init(
        project="wav2lev_final_eval",
        notes="Experiment 9: WHISP-MLP Implementation",
        tags=["exp9", "WER-GRU"],
        config=c.EXP_MODEL_CONFIG[experiment_name],
        group=experiment_name,
        name=run_name
    ) as run:
        
        torch.random.manual_seed(2378394095)
        #cuda device
        device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")
        
        train_dataset = Dataset.from_pandas(
            dd.read_parquet(f"{c.CNOISY_DATASET_PATH}/train.parquet").compute())
        validation_dataset = Dataset.from_pandas(
            dd.read_parquet(f"{c.CNOISY_DATASET_PATH}/validation.parquet").compute())
        
        char2idx, idx2char, blank_token_id = init_transformer_vocab()
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=c.EXP_MODEL_CONFIG[experiment_name]['batch_size'], 
            shuffle=True, 
            collate_fn=collate_fn_cnoisy_precomputed,
            num_workers=1,
            pin_memory=True,
            persistent_workers=True
        )
        
        val_loader = DataLoader(
            validation_dataset, 
            batch_size=c.EXP_MODEL_CONFIG[experiment_name]['batch_size'], 
            shuffle=False, 
            collate_fn=collate_fn_cnoisy_precomputed,
            num_workers=1,
            pin_memory=True,
            persistent_workers=True
        )
        
        model = model_class().to(device)
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\n\n\nTotal parameters: {total_params}")
        print(f"Trainable parameters: {trainable_params}\n\n\n")
        
        # For debug:
        # model.load_state_dict(torch.load(f'{c.WEIGHTS_PATH}/exp_6_whisper_2025-09-15_11:46:36_best.pt', map_location=device))
            
        optimizer = AdamW(
            filter(lambda p: p.requires_grad, model.parameters()), 
            lr=c.EXP_MODEL_CONFIG[experiment_name]['learning_rate'],
            weight_decay=c.EXP_MODEL_CONFIG[experiment_name]['weight_decay']
        )
        
        scaler = torch.cuda.amp.GradScaler()
    
        log.debug(f"Begin model training for {model_name}")
        
        results = train(model,
            train_loader,
            val_loader,
            optimizer,
            num_epochs=c.EXP_MODEL_CONFIG[experiment_name]['epochs'],
            device=device,
            idx2char=idx2char,
            scaler=scaler,
            log=log,
            run=run)
        
        output_dir = f'out/{experiment_name}_results'
        os.makedirs(output_dir, exist_ok=True)
        
        results_path = f'{output_dir}/{model_name}_results.csv'
        
        summary_df = pd.DataFrame([{
            'model': model_name,
            'best_val_rmse': results['best_val_rmse'],
            'final_val_rmse': results['final_val_rmse'],
            'final_val_pearson': results['final_val_pearson'],
            'timestamp': ts
        }])
        summary_df.to_csv(results_path, index=False)
        
        detailed_df = pd.DataFrame({
            'model': [model_name] * len(results['val_rmse_history']),
            'validation_step': range(len(results['val_rmse_history'])),
            'rmse': results['val_rmse_history'],
            'pearson': results['val_pearson_history'],
            'timestamp': [ts] * len(results['val_rmse_history'])
        })
        detailed_path = f'{output_dir}/{model_name}_detailed_results.csv'
        detailed_df.to_csv(detailed_path, index=False)
        
        run.save(results_path)
        run.save(detailed_path)
    
    log.complete()

def exp_9_test(model_name, weights_path):
    log = Logger(script_name="Experiment 9: WHISP-MLP Implementation")
    log.welcome()
    
    experiment_name = 'exp_9'
    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    
    models = {
        'whisper': whisper_model
    }
    
    if model_name not in models:
        raise ValueError(f"Model name {model_name} not recognized. Choose from {list(models.keys())}.")
    
    model_class = models[model_name]
    run_name = f'{experiment_name}_{model_name}_test_{ts}'
    
    with wandb.init(
        project="wav2lev_final_eval",
        notes="Experiment 9: Test Set Evaluation",
        tags=["exp9", "WER-GRU", "test"],
        config=c.EXP_MODEL_CONFIG[experiment_name],
        group=experiment_name,
        name=run_name
    ) as run:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            
        test_dataset = Dataset.from_pandas(
            dd.read_parquet(f"{c.CNOISY_DATASET_PATH}/test.parquet").compute())
        
        char2idx, idx2char, blank_token_id = init_transformer_vocab()
        
        test_loader = DataLoader(
            test_dataset, 
            batch_size=c.EXP_MODEL_CONFIG[experiment_name]['batch_size'], 
            shuffle=False, 
            collate_fn=collate_fn_cnoisy_precomputed,
            num_workers=1,
            pin_memory=True,
            persistent_workers=True
        )
        
        model = model_class().to(device)
        
        model.load_state_dict(torch.load(weights_path,  map_location=device))
        
        test(model,
             test_loader,
             device,
             idx2char,
             blank_token_id,
             log,
             run)
        
        log.complete()