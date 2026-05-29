import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("occlusion_datasets/train.csv")

bins = [-0.01, 0.1, 0.3, 1.0]
df['occ_bin'] = pd.cut(df['FaceOcclusion'], bins=bins)
df['stratify_key'] = df['gender'].astype(str) + "_" + df['occ_bin'].astype(str)

train_split, val_split = train_test_split(
    df, 
    test_size=0.20, 
    random_state=42, 
    stratify=df['stratify_key']
)

df.drop(columns=['occ_bin', 'stratify_key'], inplace=True)
df['selection'] = 'train'
df.loc[val_split.index, 'selection'] = 'val'
df.to_csv('occlusion_datasets/train_clean.csv', index=False)

print(f"Taille Train: {len(train_split)} | Taille Val: {len(val_split)}")