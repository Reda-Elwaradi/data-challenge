import traceback
import torch
from functools import partial
import config
from main import main as run_experiment

def get_cosine_scheduler(current_lr, max_epochs):
    return partial(torch.optim.lr_scheduler.CosineAnnealingLR, T_max=max_epochs, eta_min=current_lr/100)

def get_step_scheduler():
    return partial(torch.optim.lr_scheduler.StepLR, step_size=7, gamma=0.1)

def get_reduce_on_plateau_scheduler():
    return partial(torch.optim.lr_scheduler.ReduceLROnPlateau, mode='min', factor=0.1, patience=2)

experiments = [
    {
        "model_save_path": "saved_models/resnet1.pth"
    },
    {
        "model_save_path": "saved_models/resnet2.pth",
        "differential_lr": 0.5,
        "unfreeze_blocks": 4,
    },
    {
        "model_save_path": "saved_models/resnet3.pth",
        "differential_lr": 0.5,
        "lr": 1e-3,
    },
    {
        "model_save_path": "saved_models/resnet4.pth",
        "lr": 1e-3,
        "unfreeze_blocks": 1,
        "differential_lr": 1,
    },
    {
        "model_save_path": "saved_models/resnet5.pth",
        "differential_lr": 0.5,
        "unfreeze_blocks": 4,
        "lr": 1e-3,
    },
]

if __name__ == "__main__":
    print(f"Lancement de la série de {len(experiments)} expériences...")
    
    for i, exp in enumerate(experiments):
        print("\n" + "="*70)
        print(f"DÉMARRAGE EXPÉRIENCE {i+1} / {len(experiments)} : {exp['model_save_path']}")
        print("="*70)
        
        try:
            run_experiment(**exp)
        except Exception as e:
            print(f"\n❌ ERREUR CRITIQUE PENDANT L'EXPÉRIENCE {i+1} !")
            traceback.print_exc()
            
    print("\nTOUTES LES EXPÉRIENCES SONT TERMINÉES !")