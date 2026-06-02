import gc
import torch
from tqdm import tqdm
import optuna
from config import *
from data import get_train_val_loaders
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from model import Modele, IdemiaLoss

def train(model, model_retrieve_path, train_loader, val_loader, max_epochs, epoch_start, early_stopping_patience, total_batch_size, lr, weight_decay, differential_lr, save_path, unfreeze_blocks, unfreeze_epoch, scheduler_fn, gpu_train_transforms, gpu_val_transforms, trial):
    accumulation_steps = total_batch_size // BATCH_SIZE
    criterion = IdemiaLoss()
    if differential_lr is not None:
        fc_params = list(model.fc.parameters())
        base_params = [p for n, p in model.named_parameters() if not n.startswith('fc')]
        optimizer = optim.Adam([
            {'params': base_params, 'lr': lr * differential_lr}, 
            {'params': fc_params, 'lr': lr}
        ], weight_decay=weight_decay)
    else:
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = scheduler_fn(optimizer)
    unfreezed = False
    if trial is None:
        writer = SummaryWriter('runs/'+save_path.split('/')[-1].split('.')[0])
    best_val_loss = float('inf')
    patience_counter = 0
    best_epoch = 0
    start_epoch = epoch_start - 1
    if model_retrieve_path is not None:
        checkpoint = torch.load(model_retrieve_path, map_location=DEVICE)
        if 'optimizer_state_dict' in checkpoint:
            print(f"\n🔄 Reprise complète de l'entraînement depuis l'époque {checkpoint['epoch']}...")
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch']
            best_val_loss = checkpoint['best_val_loss']
            patience_counter = checkpoint['patience_counter']
            unfreezed = checkpoint.get('unfreezed', False)

    for epoch in range(start_epoch, max_epochs):
        running_loss = 0.0
        val_loss = 0.0
        total_train = 0
        if unfreeze_epoch is not None and epoch >= unfreeze_epoch and not unfreezed:
            print(f"\nDégel de {unfreeze_blocks if unfreeze_blocks is not None else model.max_blocks} blocs profonds à partir de l'époque {epoch+1}...")
            model.unfreeze_blocks(unfreeze_blocks if unfreeze_blocks is not None else model.max_blocks)
            unfreezed = True

        model.train()
        with tqdm(train_loader, desc="Entraînement") as loop:
            outputs_list, labels_list, genders_list = [], [], []
            for i, (images, labels, genders) in enumerate(loop):
                # images, labels, genders = images.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True), genders.to(DEVICE, non_blocking=True)
                images = images.to(DEVICE, non_blocking=True)
                labels = labels.to(DEVICE, dtype=torch.float32, non_blocking=True)
                genders = genders.to(DEVICE, dtype=torch.float32, non_blocking=True)
                if gpu_train_transforms:
                    images = gpu_train_transforms(images)
                outputs = model(images)
                outputs_list.append(outputs)
                labels_list.append(labels)
                genders_list.append(genders)

                if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
                    outputs = torch.cat(outputs_list, dim=0)
                    labels = torch.cat(labels_list, dim=0)
                    genders = torch.cat(genders_list, dim=0)
                    loss = criterion(outputs, labels, genders)
                    loss.backward()

                    running_loss += loss.item() * labels.size(0)
                    total_train += labels.size(0)
                    loop.set_postfix(loss=running_loss/total_train)

                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                    optimizer.step()
                    optimizer.zero_grad()
                    outputs_list.clear()
                    labels_list.clear()
                    genders_list.clear()
        
        model.eval()
        with torch.no_grad(): 
            with tqdm(val_loader, desc="Validation") as val_bar:
                outputs_list, labels_list, genders_list = [], [], []
                for i, (images, labels, genders) in enumerate(val_bar):
                    images = images.to(DEVICE, non_blocking=True)
                    labels = labels.to(DEVICE, dtype=torch.float32, non_blocking=True)
                    genders = genders.to(DEVICE, dtype=torch.float32, non_blocking=True)
                    if gpu_val_transforms:
                        images = gpu_val_transforms(images)
                    outputs = model(images)
                    outputs_list.append(outputs)
                    labels_list.append(labels)
                    genders_list.append(genders)

                outputs = torch.cat(outputs_list, dim=0)
                labels = torch.cat(labels_list, dim=0)
                genders = torch.cat(genders_list, dim=0)
                loss = criterion(outputs, labels, genders)

                val_loss = loss.item()
                print(f"Loss de validation à l'époque {epoch+1} : {val_loss:.5f}")
    
        avg_val_loss = val_loss
        if trial is not None:
            trial.report(avg_val_loss, epoch)
            if trial.should_prune():
                print(f"Essai {trial.number} élagué (pruned) à l'époque {epoch+1}.")
                raise optuna.exceptions.TrialPruned()
        avg_train_loss = running_loss / total_train
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(avg_val_loss)
        else:
            scheduler.step()
        optimizer.zero_grad()

        if differential_lr is not None:
            current_lr_backbone = optimizer.param_groups[0]['lr']
            current_lr_classifier = optimizer.param_groups[1]['lr']
            print(f"Époque {epoch+1} terminée. LR actuel (backbone) : {current_lr_backbone:<.2e}, LR actuel (classifier) : {current_lr_classifier:<.2e}")
        else:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Époque {epoch+1} terminée. LR actuel : {current_lr:<.2e}")

        if trial is None:
            writer.add_scalars('Loss', {
                'Train': avg_train_loss,
                'Validation': avg_val_loss
            }, epoch + 1)
            if differential_lr is not None:
                writer.add_scalars('Learning_Rate', {
                    'Backbone': current_lr_backbone,
                    'FC_Classifier': current_lr_classifier
                }, epoch + 1)
            else:
                writer.add_scalar('Learning_Rate', current_lr, epoch + 1)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_epoch = epoch + 1
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
                'patience_counter': patience_counter,
                'unfreezed': unfreezed
            }
            torch.save(checkpoint, save_path)
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print(f"\nEARLY STOPPING DÉCLENCHÉ : Arrêt de l'entraînement à l'époque {epoch+1}.")
                break
    print(f"\nEntraînement terminé. Meilleure époque : {best_epoch} avec une loss de validation de {best_val_loss:.5f}.")
    if trial is None:
        writer.close()
    return best_epoch, best_val_loss

def main(model_retrieve_path=MODEL_RETRIEVE_PATH, dropout_rate=DROPOUT_RATE, max_epochs=MAX_EPOCHS, epoch_start=EPOCH_START, early_stopping_patience=EARLY_STOPPING_PATIENCE, total_batch_size=TOTAL_BATCH_SIZE, lr=LR, weight_decay=WEIGHT_DECAY, differential_lr=DIFFERENTIAL_LR, model_save_path=MODEL_SAVE_PATH, unfreeze_blocks=UNFREEZE_BLOCKS, unfreeze_epoch=UNFREEZE_EPOCH, scheduler_fn=SCHEDULER_FN, image_perturbations=IMAGE_PERTURBATIONS, trial=None):
    train_loader, val_loader, t_trans, v_trans = get_train_val_loaders(image_perturbations)
    MODEL = Modele(model_retrieve_path, dropout_rate, unfreeze_epoch is not None)
    MODEL.to(DEVICE, non_blocking=True)
    if ON_GPU_TRANSFORM:
        t_trans = t_trans.to(DEVICE, non_blocking=True)
        v_trans = v_trans.to(DEVICE, non_blocking=True)
    epoch_start = epoch_start if model_retrieve_path is not None else 1
    try:
        best_epoch, best_val_loss = train(MODEL, model_retrieve_path, train_loader, val_loader, max_epochs, epoch_start, early_stopping_patience, total_batch_size, lr, weight_decay, differential_lr, model_save_path, unfreeze_blocks, unfreeze_epoch, scheduler_fn, t_trans, v_trans, trial)
        return best_epoch, best_val_loss
    finally:
        if 'MODEL' in locals():
            del MODEL
        if 'train_loader' in locals():
            del train_loader
        if 'val_loader' in locals():
            del val_loader
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

if __name__ == "__main__":
    main()