import gc
import pandas as pd
from tqdm import tqdm
import torch
from config import *
from data import get_test_loader
from model import Modele

def evaluate_and_map(model, test_loader, save_path, gpu_test_transforms):
    model.eval()
    
    print("Évaluation en cours sur le set de test...")
    with torch.no_grad():
        with tqdm(test_loader, desc="Prédictions") as loop:
            outputs_list = []
            for i, images in enumerate(loop):
                images = images.to(DEVICE, non_blocking=True)
                if gpu_test_transforms:
                    images = gpu_test_transforms(images)
                outputs = model(images)
                outputs_list.append(outputs)

            outputs = torch.cat(outputs_list, dim=0)

    path = "saved_evals/" + ".".join(save_path.replace("saved_models/", "").split(".")[:-1]) + ".csv"
    df = pd.read_csv(BASE_TEST_CSV_PATH)
    df['FaceOcclusion'] = outputs.cpu().numpy().flatten()
    df.to_csv(path, index=False)
    return

def main(model_save_path, trial=False):
    test_loader, t_trans = get_test_loader()
    MODEL = Modele(model_save_path, 0, True)
    MODEL.to(DEVICE, non_blocking=True)
    if ON_GPU_TRANSFORM:
        t_trans = t_trans.to(DEVICE, non_blocking=True)
    try:
        results = evaluate_and_map(MODEL, test_loader, model_save_path, t_trans)
        return results
    finally:
        del MODEL
        del test_loader
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

if __name__ == "__main__":
    main(MODEL_SAVE_PATH)