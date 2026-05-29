import os
import csv
from train_eval import main as train_main
from test import main as test_main
from config import *

def main(model_retrieve_path=MODEL_RETRIEVE_PATH, dropout_rate=DROPOUT_RATE, max_epochs=MAX_EPOCHS, epoch_start=EPOCH_START, early_stopping_patience=EARLY_STOPPING_PATIENCE, total_batch_size=TOTAL_BATCH_SIZE, lr=LR, weight_decay=WEIGHT_DECAY, differential_lr=DIFFERENTIAL_LR, model_save_path=MODEL_SAVE_PATH, unfreeze_blocks=UNFREEZE_BLOCKS, unfreeze_epoch=UNFREEZE_EPOCH, scheduler_fn=SCHEDULER_FN, image_perturbations=IMAGE_PERTURBATIONS):
    print("Début de l'entraînement...")
    best_epoch, best_val_loss = train_main(model_retrieve_path, dropout_rate, max_epochs, epoch_start, early_stopping_patience, total_batch_size, lr, weight_decay, differential_lr, model_save_path, unfreeze_blocks, unfreeze_epoch, scheduler_fn, image_perturbations)
    print("Entraînement terminé. Début du test...")
    test_main(model_save_path)
    print("Test terminé. Enregistrement des résultats...")

    file_exists = os.path.isfile(RESULTS_CSV_PATH)
    headers = ["Modele", "Dropout_Rate", "LR", "Weight_Decay", "Differential_LR", "Scheduler", "Epochs", "Batch_Total_Size", "Unfreeze_Blocks", "Unfreeze_Epoch", "Val loss", "Commentaires"]
    row = {
        "Modele": model_save_path.split('/')[-1].split(".")[0],
        "Dropout_Rate": dropout_rate,
        "LR": lr,
        "Weight_Decay": weight_decay,
        "Differential_LR": differential_lr,
        "Scheduler": scheduler_fn.func.__name__,
        "Epochs": best_epoch,
        "Batch_Total_Size": total_batch_size,
        "Unfreeze_Blocks": unfreeze_blocks,
        "Unfreeze_Epoch": unfreeze_epoch,
        "Val loss": best_val_loss,
        "Commentaires": ""
    }

    # Écriture dans le fichier CSV
    with open(RESULTS_CSV_PATH, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
    
    return best_val_loss

if __name__ == "__main__":
    main()