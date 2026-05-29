import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from torchvision.transforms import v2
from torchvision.io import read_image, ImageReadMode
from config import *

def get_transforms(image_perturbations=IMAGE_PERTURBATIONS):
    cpus_transforms = {
        "train": transforms.Compose([
            # 1. RandomApply : % de chance de modifier les couleurs/lumières
            transforms.RandomApply([transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05)], p=image_perturbations[0]),
            # 2. RandomApply : % de chance d'appliquer une légère rotation
            transforms.RandomApply([transforms.RandomRotation(degrees=10, interpolation=transforms.InterpolationMode.BILINEAR)], p=image_perturbations[1]),
            # 3. RandomApply : % de chance de flouter très légèrement l'image
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=9, sigma=(0.1, 2.0))], p=image_perturbations[2]),
            transforms.ConvertImageDtype(torch.float32),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        "val": transforms.Compose([
            transforms.ConvertImageDtype(torch.float32),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        "test": transforms.Compose([
            transforms.ConvertImageDtype(torch.float32),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    }
    gpu_transforms = {
        "train": v2.Compose([
            v2.ToDtype(torch.float32, scale=True),
            v2.RandomApply([v2.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05)], p=image_perturbations[0]),
            v2.RandomApply([v2.RandomRotation(degrees=10, interpolation=transforms.InterpolationMode.BILINEAR)], p=image_perturbations[1]),
            v2.RandomApply([v2.GaussianBlur(kernel_size=9, sigma=(0.1, 2.0))], p=image_perturbations[2]),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        "val": v2.Compose([
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        "test": v2.Compose([
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    }
    return cpus_transforms, gpu_transforms

class CustomDataset(Dataset):
    def __init__(self, split, cpus_transforms, gpu_transforms):
        if split == 'test':
            df = pd.read_csv(BASE_TEST_CSV_PATH)
        else:
            df = pd.read_csv(CSV_PATH)
            df = df[df['selection'] == split]
        data = df.reset_index(drop=True)
        #data = data.sample(frac=1).reset_index(drop=True)
        self.filenames = data['filename'].tolist()

        if split == 'train':
            self.labels = data['FaceOcclusion'].tolist()
            self.genders = data['gender'].tolist()
            class_counts = data['gender'].value_counts().to_dict()
            class_weights = {gender: 1.0 / count for gender, count in class_counts.items()}
            sample_weights = [class_weights[row['gender']] for _, row in data.iterrows()]
            self.sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(sample_weights),
                replacement=True
            )
            self.transform = None if ON_GPU_TRANSFORM else cpus_transforms["train"]
            print(f"Dataset de train : Total : {len(data)} images.")
        else:
            self.sampler = None
            self.transform = None if ON_GPU_TRANSFORM else cpus_transforms[split]
            self.labels = None
            self.genders = None
            print(f"Dataset de {split} : Total : {len(data)} images.")

        self.img_dir = IMAGE_PATH
        self.split = split
        
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, idx):
        img_name = self.filenames[idx]
        img_path = os.path.join(self.img_dir, img_name)
        img = read_image(img_path, mode=ImageReadMode.RGB)
        if self.transform:
            img = self.transform(img)
        
        if self.split == 'test':
            return img
        else:
            label = self.labels[idx]
            gender = self.genders[idx]
            return img, label, gender

def get_train_val_loaders(image_perturbations):
    cpus_transforms, gpu_transforms = get_transforms(image_perturbations)
    train_dataset = CustomDataset('train', cpus_transforms, gpu_transforms)
    val_dataset = CustomDataset('val', cpus_transforms, gpu_transforms)
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=train_dataset.sampler,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE_VAL,
        sampler=None,
        shuffle=False,
        num_workers=NUM_WORKERS_VAL,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )
    t_trans, v_trans = (gpu_transforms["train"], gpu_transforms["val"]) if ON_GPU_TRANSFORM else (None, None)
    print("Nombre de batchs en entraînement :", len(train_loader))
    print("Nombre de batchs en validation :", len(val_loader))
    return train_loader, val_loader, t_trans, v_trans

def get_test_loader():
    cpus_transforms, gpu_transforms = get_transforms()
    test_dataset = CustomDataset('test', cpus_transforms, gpu_transforms)
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE_VAL,
        sampler=None,
        shuffle=False,
        num_workers=NUM_WORKERS_VAL,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )
    t_trans = gpu_transforms["test"] if ON_GPU_TRANSFORM else None
    print("Nombre de batchs en test :", len(test_loader))
    return test_loader, t_trans