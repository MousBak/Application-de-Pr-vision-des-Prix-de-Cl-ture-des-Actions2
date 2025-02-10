# Prévision des Prix de Clôture des Actions

Application web interactive pour l'analyse et la prévision des prix de clôture des actions, développée avec Streamlit et Python.

## Fonctionnalités

### 1. Analyse des Données
- Chargement des données historiques des actions via Yahoo Finance
- Visualisation interactive des prix avec Plotly
- Affichage des statistiques clés (prix actuel, variation, volume)
- Support pour différents marchés (CAC 40, NASDAQ, etc.)

### 2. Analyse Technique
- Calcul et affichage des indicateurs techniques :
  - Moyennes mobiles (20, 50, 200 jours)
  - RSI (Relative Strength Index)
  - MACD (Moving Average Convergence Divergence)
- Visualisation des signaux d'achat/vente

### 3. Prévision des Prix
- Modèles de prévision disponibles :
  - SARIMA
  - Prophet
  - LSTM
- Paramètres ajustables pour chaque modèle
- Visualisation des prévisions avec intervalles de confiance

### 4. Génération de Rapports
- Rapports PDF détaillés :
  - Rapport résumé : vue d'ensemble et statistiques clés
  - Rapport technique : analyse des indicateurs techniques
  - Rapport fondamental : métriques et tendances
- Graphiques et visualisations inclus
- Téléchargement facile des rapports générés

## Installation

1. Cloner le dépôt :
```bash
git clone [URL_DU_REPO]
cd "Prévision des Prix de Clôture des Actions"
```

2. Installer les dépendances :
```bash
pip install -r requirements.txt
```

## Utilisation

1. Lancer l'application :
```bash
streamlit run forecasting_app.py
```

2. Dans l'interface :
   - Sélectionner le marché et l'action désirée
   - Choisir la période d'analyse
   - Explorer les différents onglets d'analyse
   - Générer et télécharger les rapports

## Structure du Projet

```
├── forecasting_app.py     # Application principale
├── analysis/
│   ├── technical_analysis.py    # Indicateurs techniques
│   └── forecasting_models.py    # Modèles de prévision
├── utils/
│   ├── data_loader.py          # Chargement des données
│   └── report_generator.py     # Génération des rapports
├── models/                     # Modèles entraînés
└── rapports/                  # Rapports générés
```

## Dépendances Principales

- Streamlit : Interface utilisateur
- Pandas : Manipulation des données
- Plotly : Visualisations interactives
- yfinance : Données boursières
- scikit-learn : Modèles de prévision
- Prophet : Prévisions avancées
- FPDF : Génération de rapports PDF

## Contribution

Les contributions sont les bienvenues ! N'hésitez pas à :
- Signaler des bugs
- Proposer des nouvelles fonctionnalités
- Soumettre des pull requests

## Licence

Ce projet est sous licence MIT. Voir le fichier `LICENSE` pour plus de détails.
