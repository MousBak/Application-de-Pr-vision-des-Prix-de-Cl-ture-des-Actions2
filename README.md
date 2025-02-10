# Application de Prévision des Prix de Clôture des Actions

Cette application Streamlit permet de prévoir les prix de clôture des actions pour différents indices boursiers (S&P500, DAX, Nikkei225, FTSE100, CAC40) en utilisant le modèle Prophet de Facebook.

## Fonctionnalités

- Sélection d'indices boursiers multiples (S&P500, DAX, Nikkei225, FTSE100, CAC40)
- Téléchargement automatique des données historiques via Yahoo Finance
- Visualisation interactive des prix historiques
- Prévision des prix futurs avec Prophet
- Analyse des points de changement (changepoints)
- Support des jours fériés pour différents pays
- Téléchargement des modèles entraînés

## Installation

1. Cloner le repository :
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
   - Sélectionner un indice boursier
   - Choisir une action dans la liste
   - Définir la période d'analyse
   - Ajuster les paramètres de prévision si nécessaire
   - Cliquer sur "Run Forecasting"

## Structure du Projet

- `forecasting_app.py` : Application principale Streamlit
- `utils.py` : Fonctions utilitaires pour le chargement et le traitement des données
- `requirements.txt` : Liste des dépendances Python
- `.streamlit/config.toml` : Configuration Streamlit

## Paramètres de Prévision

- **Testing Data Percentage** : Pourcentage des données utilisées pour le test (0.1 à 0.4)
- **Changepoint Range** : Flexibilité des points de changement (0.05 à 0.95)
- **Country Holidays** : Jours fériés à prendre en compte (US, FR, DE, JP, GB)
- **Forecast Horizon** : Nombre de jours à prévoir

## Dépendances Principales

- Streamlit 1.12.0
- Pandas 1.5.3
- Prophet 1.1.4
- YFinance 0.2.18
- Plotly 5.14.1

## Remarques

- L'application nécessite une connexion Internet pour télécharger les données
- Les prévisions sont basées sur les données historiques et ne garantissent pas les performances futures
- Certains symboles boursiers peuvent ne pas être disponibles selon la région

## Support

Pour toute question ou problème, veuillez ouvrir une issue dans le repository GitHub.
