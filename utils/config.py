
class Config():

    CACHE_PATH = "./mini_cnoisy/cache"
    WEIGHTS_PATH = "./mini_cnoisy/weights"
    CNOISY_DATASET_PATH = "./mini_cnoisy/data/cnoisy_final_v3"
    EMBEDDINGS_PRECOMP_PATH = "./whisp_embeddings"
    WHISPER_MODEL_ID = "openai/whisper-large-v3"
    # Max Levenshtein sequence length stored in data (includes <start>/<end> tokens)
    MAX_EDIT_SEQ_LEN = 400

    LEVTRANSDUCER_MODEL_CONFIG = {
        "epochs": 40,
        "learning_rate": 1e-5,
        "batch_size": 4,
        "weight_decay": 1e-4,
    }

    LEVCTC_MODEL_CONFIG = {
        "epochs": 500,
        "learning_rate": 1e-3,
        "batch_size": 32,
        "weight_decay": 1e-4,
    }

    LEVTRANSFORMER_MODEL_CONFIG = {
        "epochs": 100,
        "learning_rate": 12e-6,
        "batch_size": 32,
        "weight_decay": 1e-4,
        # architecture
        "hidden_size": 1024,
        "num_decoder_layers": 12,
        "num_heads": 16,
        "dropout": 0.1,
        "max_edit_length": 300,
        # training
        "label_smoothing": 0.05,
        "warmup_ratio": 0.01,
        "gradient_accumulation_steps": 2,
        "max_grad_norm": 1.0,
    }

    WERGRU_MODEL_CONFIG = {
        "epochs": 100,
        "learning_rate": 12e-6,
        "batch_size": 16,
        "weight_decay": 1e-4,
        "warmup_ratio": 0.01,
        "gradient_accumulation_steps": 2,
        "max_grad_norm": 1.0,
    }

    WHISP_MLP_MODEL_CONFIG = {
        "epochs": 100,
        "learning_rate": 12e-6,
        "batch_size": 64,
        "weight_decay": 1e-4,
        "warmup_ratio": 0.01,
        "gradient_accumulation_steps": 2,
        "max_grad_norm": 1.0,
    }

    EXP_MODEL_CONFIG = {
        "exp_1": LEVTRANSDUCER_MODEL_CONFIG,
        "exp_2": LEVTRANSDUCER_MODEL_CONFIG,
        "exp_3": LEVTRANSDUCER_MODEL_CONFIG,
        "exp_4": LEVCTC_MODEL_CONFIG,
        "exp_5": LEVTRANSFORMER_MODEL_CONFIG,
        "exp_6": WHISP_MLP_MODEL_CONFIG,
        "exp_7": LEVTRANSFORMER_MODEL_CONFIG,
        "exp_8": WERGRU_MODEL_CONFIG,
        "exp_9": WHISP_MLP_MODEL_CONFIG,
        "exp_10": LEVTRANSFORMER_MODEL_CONFIG,
    }
