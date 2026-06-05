import gc
import os
import psutil
import csv
from functools import partial
import torch
import torch.optim.lr_scheduler as lr_scheduler
import optuna
from optuna.visualization import plot_optimization_history, plot_parallel_coordinate, plot_param_importances, plot_contour

from train_eval import main as train_main
from test import main as test_main
from config import *

def get_scheduler(trial, lr):
    scheduler_name = trial.suggest_categorical(
        "scheduler", ["CosineAnnealingLR", "StepLR", "ReduceLROnPlateau"]
    )
    
    if scheduler_name == "CosineAnnealingLR":
        max_epochs = trial.suggest_int("max_epochs", 20, 40, step=5)
        scheduler_params = {"T_max": max_epochs, "eta_min": lr/100}
        return max_epochs, partial(lr_scheduler.CosineAnnealingLR, T_max=max_epochs, eta_min=lr/100), scheduler_name, scheduler_params
    else:
        max_epochs = MAX_EPOCHS
    if scheduler_name == "StepLR":
        step_size = trial.suggest_int("step_size", 5, 20)
        gamma = trial.suggest_float("gamma", 0.1, 0.5)
        scheduler_params={"step_size":step_size,"gamma": gamma}
        return max_epochs, partial(lr_scheduler.StepLR, step_size=step_size, gamma=gamma), scheduler_name, scheduler_params
    elif scheduler_name == "ReduceLROnPlateau":
        factor = trial.suggest_float("factor", 0.1, 0.5)
        patience = trial.suggest_int("patience", 1, 5)
        scheduler_params={"factor":factor,"patience": patience}
        return max_epochs, partial(lr_scheduler.ReduceLROnPlateau, mode='min', factor=factor, patience=patience), scheduler_name, scheduler_params

def objective(trial):
    model_save_path = f"saved_models/optuna_global/trial_{trial.number+1}.pth"

    total_batch_size = trial.suggest_categorical("total_batch_size", [128, 256])
    lr = trial.suggest_float("learning_rate", 1e-4, 1e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)
    differential_lr = trial.suggest_float("differential_lr", 0.3, 1)
    dropout_rate = trial.suggest_float("dropout", 0.1, 0.5, step=0.1)
    unfreeze_blocks = trial.suggest_int("unfreeze_blocks", 0, 4)
    unfreeze_epoch = trial.suggest_int("unfreeze_epoch", 0, 5)
    max_epochs, scheduler_fn, scheduler_name, scheduler_params = get_scheduler(trial, lr)

    print(
        f"Essai {trial.number+1}:\n"
        f"Max epoch = {max_epochs}, Scheduler = {scheduler_name}, Scheduler_Params = {scheduler_params}\n"
        f"lr={lr:.4f}, weight_decay={weight_decay:.4f},  differential_lr={differential_lr:.4f},\n" 
        f"unfreeze_blocks={unfreeze_blocks},unfreeze_epoch={unfreeze_epoch}, dropout_rate={dropout_rate:.3f}",
    )

    try:
        best_epoch, best_val_loss = train_main(
            dropout_rate=dropout_rate,
            max_epochs=max_epochs,
            total_batch_size=total_batch_size,
            lr=lr,
            weight_decay=weight_decay,
            differential_lr=differential_lr,
            model_save_path=model_save_path,
            unfreeze_blocks=unfreeze_blocks,
            unfreeze_epoch=unfreeze_epoch,
            scheduler_fn=scheduler_fn,
            trial=trial
        )
        print(f"\nTest du modèle : {model_save_path}")
        test_main(model_save_path=model_save_path)

        trial.set_user_attr("best_epoch", best_epoch)
        trial.set_user_attr("val_loss", best_val_loss)
        p = trial.params
        row = {
            "Modele": f"Optuna_global_Trial_{trial.number+1}",
            "Type": MODEL_NAME,
            "Dropout_Rate": p.get("dropout", DROPOUT_RATE),
            "LR": p.get("learning_rate", LR),
            "Weight_Decay": p.get("weight_decay", WEIGHT_DECAY),
            "Differential_LR": p.get("differential_lr", DIFFERENTIAL_LR),
            "Scheduler": p.get("scheduler", SCHEDULER_FN.func.__name__),
            "Epochs": best_epoch,
            "Batch_Total_Size": p.get("total_batch_size", TOTAL_BATCH_SIZE),
            "Unfreeze_Blocks": p.get("unfreeze_blocks", UNFREEZE_BLOCKS),
            "Unfreeze_Epoch": p.get("unfreeze_epoch", UNFREEZE_EPOCH),
            "Val loss": round(best_val_loss, 5),
            "Commentaires": f"Trial={trial.number + 1}"
        }

        file_exists = os.path.isfile(RESULTS_CSV_PATH)
        headers = ["Modele", "Type", "Dropout_Rate", "LR", "Weight_Decay", "Differential_LR", "Scheduler", "Epochs", "Batch_Total_Size", "Unfreeze_Blocks", "Unfreeze_Epoch", "Val loss", "Commentaires"]
        with open(RESULTS_CSV_PATH, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)
    except optuna.exceptions.TrialPruned:
        raise optuna.exceptions.TrialPruned()
    except Exception as e:
        print(f"❌ Erreur durant l'essai: {e}")
        raise e
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        try:
            parent = psutil.Process(os.getpid())
            enfants = parent.children(recursive=True)
            for enfant in enfants:
                enfant.terminate()
            gone, alive = psutil.wait_procs(enfants, timeout=3.0)
            for p in alive:
                print(f"⚠️ Processus zombie {p.pid} tué de force.")
                p.kill()
                
        except psutil.NoSuchProcess:
            pass
        except Exception as e:
            print(f"Erreur lors du nettoyage : {e}")

    return best_val_loss


def optimize_hyperparameters(n_trials, study_name):
    os.makedirs("saved_models/optuna_global", exist_ok=True)
    os.makedirs("saved_evals/optuna_global", exist_ok=True)
    pruner = optuna.pruners.HyperbandPruner(
        min_resource=4,
        max_resource=50,
        reduction_factor=3
    )
    study = optuna.create_study(
        study_name=study_name,
        direction="minimize",
        pruner=pruner,
        storage="sqlite:///optuna_geoguessr.db",
        load_if_exists=True
    )

    print(f"Démarrage de l'optimisation sur {n_trials} essais...")
    study.optimize(objective, n_trials=n_trials)

    print("OPTIMISATION TERMINEE")
    fig_hist = plot_optimization_history(study)
    fig_hist.write_html("saved_evals/optuna_global/1_optimization_history.html")
    fig_importances = plot_param_importances(study)
    fig_importances.write_html("saved_evals/optuna_global/2_param_importances.html")
    fig_contour = plot_contour(study)
    fig_contour.write_html("saved_evals/optuna_global/3_contour_lr_wd.html")
    fig_parallel = plot_parallel_coordinate(study)
    fig_parallel.write_html("saved_evals/optuna_global/4_parallel_coordinate.html")

    print(f"\nTous les modèles terminés ont été évalués et ajoutés à {RESULTS_CSV_PATH}.")

    return study


if __name__ == "__main__":
    study = optimize_hyperparameters(n_trials=200, study_name="data-challenge-global")