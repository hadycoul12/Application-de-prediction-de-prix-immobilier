# 🏠 ImmoPredict — Application Streamlit

**Auteur** : Hady COULIBALY — Étudiant Data Science  
**Version** : 2.0 — Déploiement Streamlit Cloud  

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io)
![Python](https://img.shields.io/badge/Python-3.10+-blue)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📋 Description

Application de **prédiction de prix immobilier** basée sur le dataset **Ames Housing**.  
Construite avec Streamlit, scikit-learn, et conforme aux bonnes pratiques RGPD.

---

## 🗂️ Structure du projet

```
Projet_streamlit/
│
├── app.py                          ← Application principale (sécurisée)
├── requirements.txt                ← Dépendances Python
├── README.md                       ← Ce fichier
├── .gitignore                      ← Fichiers exclus du dépôt
│
├── .streamlit/
│   ├── config.toml                 ← Thème + sécurité serveur
│   └── secrets.toml                ← 🔒 NE PAS COMMITER (gitignore)
│
└── train.csv                       ← Dataset Ames Housing (optionnel)
```

---

## 🚀 Installation locale

### 1. Cloner le dépôt

```bash
git clone https://github.com/votre-username/Projet_streamlit.git
cd Projet_streamlit
```

### 2. Créer l'environnement virtuel

```bash
python -m venv venv

# Windows CMD
venv\Scripts\activate.bat

# Mac / Linux
source venv/bin/activate
```

### 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

### 4. Configurer les secrets en local

Créer le fichier `.streamlit/secrets.toml` :

```toml
[auth]
username = "admin"
password = "votre_mot_de_passe"
```

> ⚠️ Ce fichier est dans le `.gitignore` — il ne sera **jamais** poussé sur GitHub.

### 5. Lancer l'application

```bash
streamlit run app.py
```

Ouvre **http://localhost:8501** dans votre navigateur.

---

## ☁️ Déploiement sur Streamlit Cloud

### Étape 1 — Pousser sur GitHub

```bash
git init
git add app.py requirements.txt .streamlit/config.toml .gitignore README.md
# ⚠️ NE PAS ajouter secrets.toml
git commit -m "🚀 Initial deploy - ImmoPredict"
git branch -M main
git remote add origin https://github.com/votre-username/Projet_streamlit.git
git push -u origin main
```

### Étape 2 — Créer l'app sur Streamlit Cloud

1. Aller sur **https://share.streamlit.io**
2. Cliquer **"New app"**
3. Sélectionner votre dépôt GitHub
4. Choisir `main` comme branche
5. Définir `app.py` comme fichier principal
6. Cliquer **"Deploy!"**

### Étape 3 — Configurer les secrets

1. Dans votre app Streamlit Cloud → **Settings** → **Secrets**
2. Coller :

```toml
[auth]
username = "admin"
password = "votre_mot_de_passe_fort"
```

3. Cliquer **Save** → l'app redémarre automatiquement

### Étape 4 — Vérifier le déploiement

- ✅ URL publique : `https://votre-app.streamlit.app`
- ✅ Cadenas 🔒 visible dans le navigateur (HTTPS)
- ✅ Page de connexion s'affiche
- ✅ Login fonctionne avec vos credentials

---

## 🔒 Sécurité

| Mesure | Description |
|--------|-------------|
| 🔒 HTTPS | Fourni automatiquement par Streamlit Cloud |
| 🔐 Authentification | Login requis, credentials hashés SHA-256 |
| 🛡️ Validation entrées | Nettoyage XSS, limites numériques, max longueur |
| 📋 Logging | Chaque action horodatée (connexion, upload, prédiction) |
| 🚫 Upload limité | Max 10 Mo · Max 10 000 lignes |
| 🔑 Secrets externalisés | `st.secrets` — jamais dans le code |
| 🔄 XSRF | `enableXsrfProtection = true` |
| 👁️ Pas de tracking | `gatherUsageStats = false` |

---

## 📄 Pages de l'application

### 📂 Page 1 — Upload & Exploration
- Chargement CSV sécurisé avec validation
- 5 filtres interactifs (prix, quartier, qualité, surface, zone)
- Histogrammes, scatter plot, boxplot, matrice de corrélation
- Téléchargement des données filtrées

### 🤖 Page 2 — Entraînement & Performances
- 3 modèles : Régression Linéaire, Random Forest, Gradient Boosting
- Métriques : R², MAE, RMSLE, Cross-Validation 5 folds
- Graphiques : comparaison, réel vs prédit, résidus

### 🔮 Page 3 — Prédiction
- Formulaire de saisie avec validation des entrées
- Estimation du prix avec fourchette ±10%
- Positionnement dans la distribution du marché

### 📋 Page 4 — Logs & Sécurité
- Journal d'activité de la session
- Checklist de sécurité
- Export des logs

---

## 📦 Dépendances

```
streamlit>=1.32.0
scikit-learn>=1.4.0
numpy>=1.26.0
pandas>=2.2.0
matplotlib>=3.8.0
seaborn>=0.13.0
joblib>=1.3.0
```

---

## 📊 Dataset

- **Source** : Ames Housing Dataset (Kaggle / sklearn)
- **1 460 observations** · **80 variables**
- **Cible** : `SalePrice` (prix de vente en dollars)

---

## 🤝 Contribution

Projet réalisé dans le cadre d'un exercice académique de Data Science.  
**Auteur** : Hady COULIBALY
