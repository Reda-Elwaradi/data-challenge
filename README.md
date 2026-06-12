# 🎭 Face Occlusion Detection - Data Challenge

Ce dépôt contient le code source de l'équipe **Group 13** (Évariste Parmentier, Reda Elwaradi, Zaher Hamadeh) pour le Data Challenge de détection d'occlusion faciale de Télécom Paris. 

Notre solution finale repose sur une architecture **EfficientNet-B0**, optimisée pour gérer les formats d'images complexes (`.webp`) via un pipeline de données tolérant aux pannes, et entraînée sur le cluster GPU SLURM de l'école.

Notre meilleur modèle a atteint un score de **0.00146** sur le Leaderboard public.

---

## 📁 Architecture du Projet

Le code a été pensé pour être modulaire, séparant la configuration, le traitement des données et la logique d'entraînement :

```text
📦 data-challenge
 ┣ 📂 logs/                 # Fichiers de sortie SLURM (.out et .err)
 ┣ 📂 occlusion_datasets/   # Dossier contenant les données (train, val, test)
 ┣ 📂 saved_evals/          # Prédictions générées sous format .csv
 ┣ 📂 saved_models/         # Poids des modèles entraînés (.pth)
 ┣ 📜 config.py             # Hyperparamètres et chemins d'accès globaux
 ┣ 📜 data.py               # Dataset customisé (PIL) et DataLoaders
 ┣ 📜 model.py              # Définition de l'architecture (EfficientNet, Swin)
 ┣ 📜 train_eval.py         # Boucle d'entraînement, validation et Early Stopping
 ┣ 📜 test.py               # Inférence sur le jeu de test et génération du CSV
 ┣ 📜 run_optuna.py         # Script d'optimisation des hyperparamètres
 ┣ 📜 run.sh                # Script de soumission pour le cluster SLURM
 ┗ 📜 requirements.txt      # Dépendances Python
```

## ⚙️ Installation et Prérequis

Ce projet utilise PyTorch et a été conçu pour tourner sur un environnement GPU (NVIDIA RTX 3090).

**1. Cloner le dépôt :**
```bash
git clone [https://gitlab.telecom-paris.fr/relwaradi-25/data-challenge.git](https://gitlab.telecom-paris.fr/relwaradi-25/data-challenge.git)
cd data-challenge
```

**2. Installer les dépendances :**
```bash
pip install -r requirements.txt
```
*(Assurez-vous que les datasets sont bien placés dans le dossier `occlusion_datasets/` comme défini dans `config.py`).*

## 🚀 Utilisation

### 1. Configuration
Tous les hyperparamètres (Learning Rate, Batch Size, choix du modèle) se trouvent dans le fichier `config.py`. Par défaut, le modèle est configuré sur `efficientnet_b0`.

### 2. Entraînement sur SLURM
Pour lancer l'entraînement sur le cluster de l'école avec allocation GPU :
```bash
sbatch run.sh
```
Vous pouvez suivre l'avancée de l'entraînement en direct via les fichiers générés dans le dossier `logs/` :
```bash
tail -f logs/train_<JOB_ID>.out
```
*Note : Le modèle utilise un mécanisme d'Early Stopping. Le meilleur modèle sera automatiquement sauvegardé dans `saved_models/`.*

### 3. Inférence et Génération des Prédictions
Une fois le modèle entraîné, générez le fichier `.csv` pour la soumission sur le Leaderboard en lançant :
```bash
python test.py
```
Le fichier de résultats contenant les identifiants des images et leurs scores d'occlusion sera sauvegardé dans le dossier `saved_evals/`.

## 📊 Choix Techniques (Aperçu)

* **Robustesse des Données :** Remplacement de `torchvision.io` par `PIL` dans `data.py` pour éviter les crashs de chargement liés aux headers corrompus des fichiers `.webp`.
* **Régularisation Naturelle :** Maintien des hyperparamètres par défaut de l'EfficientNet-B0 afin de contrer le *Distribution Shift* important observé entre le jeu de validation local (20 000 images) et le jeu de test caché (30 000 images).

Pour une analyse détaillée de notre méthodologie, des biais algorithmiques et de nos résultats complets, veuillez consulter notre **rapport au format PDF** soumis sur Moodle.

---
**Auteurs :** Group 13  
**École :** Télécom Paris (Juin 2026)
