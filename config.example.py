import torch
from functools import partial

CSV_PATH = 'occlusion_datasets/train_clean.csv'
BASE_TEST_CSV_PATH = 'test_predictions.csv'
DROPOUT_RATE = 0.2

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_WORKERS = 2
NUM_WORKERS_VAL = 4
IMAGE_PATH = 'images/Crop_224_5fp_100K/'
IMAGE_PERTURBATIONS = [0.3, 0.1, 0.2] # jitter, rotation, blur
ON_GPU_TRANSFORM = True

MAX_EPOCHS = 1
UNFREEZE_BLOCKS = None
UNFREEZE_EPOCH = 0

LR = 1e-4
# SCHEDULER_FN = partial(torch.optim.lr_scheduler.CosineAnnealingLR, T_max=MAX_EPOCHS, eta_min=LR/100)
# SCHEDULER_FN = partial(torch.optim.lr_scheduler.StepLR, step_size=7, gamma=0.1)
SCHEDULER_FN = partial(torch.optim.lr_scheduler.ReduceLROnPlateau, mode='min', factor=0.1, patience=2)
EARLY_STOPPING_PATIENCE = 3
WEIGHT_DECAY = 0
DIFFERENTIAL_LR = 1.0

BATCH_SIZE = 64
TOTAL_BATCH_SIZE = 64
ACCUMULATION_STEPS = TOTAL_BATCH_SIZE // BATCH_SIZE
BATCH_SIZE_VAL = 128

MODEL_RETRIEVE_PATH = None
EPOCH_START = 1
MODEL_SAVE_PATH = 'saved_models/baseline.pth'
RESULTS_CSV_PATH = 'results.csv'