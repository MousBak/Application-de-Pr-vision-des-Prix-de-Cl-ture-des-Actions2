"""
Application de prévision des prix de clôture des actions.

Cette application permet d'analyser et de prédire les prix des actions en utilisant
différentes méthodes d'analyse technique et des modèles de prévision.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import yfinance as yf
import traceback
import os

from analysis.technical_analysis import TechnicalAnalysis
from utils.report_generator import ReportGenerator

# Configuration de la page
st.set_page_config(
    page_title="Prévision des Prix de Clôture des Actions",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dictionnaire des indices boursiers
MARKET_INDICES = {
    "CAC 40": "^FCHI",
    "NASDAQ": "^IXIC",
    "S&P 500": "^GSPC",
    "Dow Jones": "^DJI"
}

def load_stock_data(symbol: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
    """
    Charge les données historiques d'une action depuis Yahoo Finance.
    
    Args:
        symbol (str): Symbole de l'action
        start_date (datetime): Date de début
        end_date (datetime): Date de fin
        
    Returns:
        pd.DataFrame: DataFrame contenant les données de l'action
    """
    try:
        stock = yf.Ticker(symbol)
        data = stock.history(start=start_date, end=end_date)
        
        # Renommer les colonnes en français
        column_mapping = {
            'Open': 'Ouverture',
            'High': 'Haut',
            'Low': 'Bas',
            'Close': 'Fermeture',
            'Volume': 'Volume',
            'Dividends': 'Dividendes',
            'Stock Splits': 'Fractionnement des actions'
        }
        data = data.rename(columns=column_mapping)
        
        return data
        
    except Exception as e:
        st.error(f"Erreur lors du chargement des données : {str(e)}")
        return None

def plot_stock_data(data: pd.DataFrame, title: str = "Évolution des prix") -> go.Figure:
    """
    Crée un graphique en chandelier des prix de l'action.
    
    Args:
        data (pd.DataFrame): Données de l'action
        title (str): Titre du graphique
        
    Returns:
        go.Figure: Figure Plotly
    """
    fig = go.Figure(data=[go.Candlestick(
        x=data.index,
        open=data['Ouverture'],
        high=data['Haut'],
        low=data['Bas'],
        close=data['Fermeture']
    )])
    
    fig.update_layout(
        title=title,
        yaxis_title='Prix',
        xaxis_title='Date',
        template='plotly_white'
    )
    
    return fig

def generate_reports(symbol: str, data: pd.DataFrame) -> None:
    """
    Génère les différents rapports d'analyse.
    
    Args:
        symbol (str): Symbole de l'action
        data (pd.DataFrame): Données de l'action
    """
    try:
        # Vérifier les données
        if data is None:
            raise ValueError("Les données ne sont pas disponibles")
        if not isinstance(data, pd.DataFrame):
            raise ValueError(f"Type de données invalide : {type(data)}")
        if len(data) == 0:
            raise ValueError("Aucune donnée disponible")
        
        # Créer une instance de ReportGenerator
        report_gen = ReportGenerator(data=data, symbol=symbol)
        
        # Générer les rapports
        reports = []
        
        try:
            summary_report = report_gen.generate_summary_report()
            if summary_report:
                reports.append(("Rapport résumé", summary_report))
        except Exception as e:
            print(f"Erreur lors de la génération du rapport résumé : {str(e)}")
        
        try:
            technical_report = report_gen.generate_technical_report()
            if technical_report:
                reports.append(("Rapport technique", technical_report))
        except Exception as e:
            print(f"Erreur lors de la génération du rapport technique : {str(e)}")
        
        try:
            fundamental_report = report_gen.generate_fundamental_report()
            if fundamental_report:
                reports.append(("Rapport fondamental", fundamental_report))
        except Exception as e:
            print(f"Erreur lors de la génération du rapport fondamental : {str(e)}")
        
        # Afficher les rapports générés
        if reports:
            st.success("✅ Rapports générés avec succès !")
            for title, path in reports:
                if os.path.exists(path):
                    with open(path, "rb") as f:
                        st.download_button(
                            label=f"Télécharger {title}",
                            data=f,
                            file_name=os.path.basename(path),
                            mime="application/pdf"
                        )
                else:
                    st.warning(f"⚠️ Le fichier {title} n'a pas été trouvé")
        else:
            st.warning("⚠️ Aucun rapport n'a pu être généré")
        
    except ValueError as ve:
        st.error(f"❌ Erreur de validation : {str(ve)}")
        print(f"Erreur de validation détaillée : {str(ve)}")
    except Exception as e:
        st.error(f"❌ Erreur lors de la génération des rapports : {str(e)}")
        print(f"Erreur détaillée : {str(e)}")
        print(f"Traceback : {traceback.format_exc()}")

def main():
    """Fonction principale de l'application."""
    
    # Sidebar pour la sélection des paramètres
    with st.sidebar:
        st.title("⚙️ Paramètres")
        
        # Sélection du marché
        market_index = st.selectbox(
            "Sélectionner un marché",
            list(MARKET_INDICES.keys())
        )
        
        # Sélection de la période
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                "Date de début",
                datetime.now() - timedelta(days=365)
            )
        with col2:
            end_date = st.date_input(
                "Date de fin",
                datetime.now()
            )
        
        # Bouton pour charger les données
        if st.button("Charger les données"):
            with st.spinner("Chargement des données..."):
                try:
                    # Charger les données de l'indice
                    symbol = MARKET_INDICES[market_index]
                    data = load_stock_data(symbol, start_date, end_date)
                    
                    if data is not None and not data.empty:
                        st.session_state.data = data
                        st.session_state.symbol = symbol
                        st.success("Données chargées avec succès!")
                    else:
                        st.error("Impossible de charger les données.")
                        
                except Exception as e:
                    st.error(f"Erreur : {str(e)}")
    
    # Corps principal
    st.title("📊 Analyse et Prévision des Actions")
    
    if 'data' in st.session_state:
        # Onglets pour différentes analyses
        tab1, tab2, tab3 = st.tabs(["📈 Prix", "📊 Analyse Technique", "📑 Rapports"])
        
        # Onglet Prix
        with tab1:
            st.subheader("Évolution des Prix")
            fig = plot_stock_data(st.session_state.data)
            st.plotly_chart(fig, use_container_width=True)
        
        # Onglet Analyse Technique
        with tab2:
            st.subheader("Indicateurs Techniques")
            
            # Créer une instance de TechnicalAnalysis
            tech_analyzer = TechnicalAnalysis(st.session_state.data)
            
            # Afficher les moyennes mobiles
            st.write("### Moyennes Mobiles")
            ma_20 = tech_analyzer.calculate_ma(window=20)
            ma_50 = tech_analyzer.calculate_ma(window=50)
            ma_200 = tech_analyzer.calculate_ma(window=200)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("MA20", f"{ma_20:.2f}")
            with col2:
                st.metric("MA50", f"{ma_50:.2f}")
            with col3:
                st.metric("MA200", f"{ma_200:.2f}")
            
            # Afficher le RSI
            st.write("### RSI")
            rsi = tech_analyzer.calculate_rsi()
            st.metric("RSI", f"{rsi:.2f}")
            
            # Afficher le MACD
            st.write("### MACD")
            macd_data = tech_analyzer.calculate_macd()
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("MACD", f"{macd_data['macd']:.2f}")
            with col2:
                st.metric("Signal", f"{macd_data['signal']:.2f}")
            with col3:
                st.metric("Histogramme", f"{macd_data['histogram']:.2f}")
        
        # Onglet Rapports
        with tab3:
            st.subheader("Génération de Rapports")
            
            if st.button("Générer les rapports"):
                if 'data' not in st.session_state:
                    st.error("Aucune donnée n'est disponible. Veuillez d'abord charger les données.")
                    return
                
                data = st.session_state.data
                symbol = st.session_state.symbol
                
                if data is None or len(data) == 0:
                    st.error("Les données sont vides ou non valides.")
                    return
                
                generate_reports(symbol, data)

if __name__ == "__main__":
    main()
