from experiments import exp_8, exp_8_test
from experiments import exp_9, exp_9_test
from experiments import exp_10, exp_10_test
from scripts import precompute_embeddings
import argparse
from utils.config import Config

c = Config()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp',
                        type=int,
                        default=10,
                        help='Experiment #')
    parser.add_argument('--model',
                        type=str,
                        default='whisper',
                        help='Model architecture name')
    parser.add_argument('--test',
                        action='store_true',
                        default=False,
                        help='Whether to run evaluation on test set')
    parser.add_argument('--precomp_embeddings',
                        action='store_true',
                        default=False,
                        help='Whether to run the precompute whisper embeddings script')
    parser.add_argument('--weights',
                        type=str,
                        default=f'exp_10_whisper_2025-12-10_23:49:10_recent.pt',
                        help='Path to model weights')
    opt = parser.parse_args()
    
    experiments = [None, None, None, None, None, None, None, None, exp_8, exp_9, exp_10]
    
    experiment_tests = [None, None, None,
                        None, None, None, 
                        None, None, exp_8_test, exp_9_test, exp_10_test]
    
    if opt.exp == 0 or opt.exp >= len(experiments):
        print(f'ERROR: experiment #{opt.exp} does not exist.')
    else:
        if opt.test:
            experiment_tests[opt.exp](opt.model, f'{c.WEIGHTS_PATH}/{opt.weights}')
        elif opt.precomp_embeddings:
            precompute_embeddings()
        else:
            experiments[opt.exp](opt.model)
        