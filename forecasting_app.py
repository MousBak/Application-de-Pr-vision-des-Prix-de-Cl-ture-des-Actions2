import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import pytz
from plotly import graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from prophet.serialize import model_to_json
from pathlib import Path
import cufflinks as cf
import os

# Import des modules personnalisés
from utils.market_components import (
    get_sp500_components,
    get_cac40_components,
    get_dax_components,
    get_ftse_components,
    get_nikkei225_components
)
from analysis.technical_analysis import TechnicalAnalysis
from analysis.fundamental_analysis import FundamentalAnalysis
from models.model_factory import ModelFactory, evaluate_model
from models.portfolio import Portfolio
from utils.report_generator import ReportGenerator
from utils.utils import (
    load_data,
    format_ticker_symbol,
    display_data_preview,
    convert_df_to_csv
)

# Configuration de la page
st.set_page_config(
    page_title="Prévision des Prix de Clôture des Actions",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Titre principal
st.title("Prévision des Prix de Clôture")

# set offline mode for cufflinks
cf.go_offline()

# sidebar
# inputs for dowloading data
st.sidebar.header("Stock Parameters")

# update available tickers based on market index selection
market_index = st.sidebar.selectbox(
    " Market Index",
      ["S&P500", "DAX", "Nikkei225", "FTSE100", "CAC40"]
)

if market_index == "S&P500":
  available_tickers, tickers_companies_dict = get_sp500_components()

elif market_index == "DAX":
  available_tickers, tickers_companies_dict = get_dax_components()

elif market_index == "Nikkei225":
  available_tickers, tickers_companies_dict = get_nikkei225_components()

elif market_index == "FTSE100":
  available_tickers, tickers_companies_dict = get_ftse_components()

elif market_index == "CAC40":
  available_tickers, tickers_companies_dict = get_cac40_components()


# available_tickers, tickers_companies_dict = get_sp500_components()

ticker = st.sidebar.selectbox(

    "Ticker",
    available_tickers,
    format_func= tickers_companies_dict.get
    )

start_date = st.sidebar.date_input(
    "Start Date",
    value=datetime(2015, 1, 1),
    min_value=datetime(2010, 1, 1),
    max_value=datetime.today()
)

end_date = st.sidebar.date_input(
    "End Date",
    value=datetime.today(),
    min_value=start_date,
    max_value=datetime.today()
)

if start_date >= end_date:
    st.sidebar.error("La date de fin doit être postérieure à la date de début")
    st.stop()

# input for technical analysis
st.sidebar.header("Forecasting Process")

exp_prophet = st.sidebar.expander("Prophet Parameters") # Renamed expander variable
test_data_percentage = exp_prophet.number_input("Testing Data Percentage", 0.1, 0.4, 0.2, 0.05)
changepoints_range = exp_prophet.number_input("Changepoint Range", 0.05, 0.95, 0.9, 0.05) # Fixed variable name
country_holidays = exp_prophet.selectbox("Country Holidays", ['US', 'FR', 'DE', 'JP', 'GB']) # Fixed variable name
horizon = exp_prophet.number_input("Forecast Horizon (days)", min_value=1, value=365, step=1) # Fixed variable name
download_prophet = exp_prophet.checkbox(label="Download Model") # Fixed variable name


# st.subheader("Modeling Process")
modeling_option = st.sidebar.radio("Select Modeling Process", ["Prophet"])

# main body

run_button = st.sidebar.button("Run Forecasting")
# Move loading data outside the 'if run_button' block
st.write(f"Tentative de chargement des données pour {ticker} du {start_date} au {end_date}")
df = load_data(ticker, start_date, end_date, market_index)

# Afficher des informations sur les données chargées
if df is not None and not df.empty:
    st.write(f"Données chargées : {len(df)} lignes")
    st.write(f"Première date : {df.index[0]}")
    st.write(f"Dernière date : {df.index[-1]}")
    
    if run_button:
        # data preview part
        def display_data_preview(title, df, fil_name, key):
            st.subheader(title)
            st.write(df.head())
            
            # Convertir le DataFrame en CSV pour le téléchargement
            csv = df.to_csv(index=True)
            st.download_button(
                label="Télécharger les données",
                data=csv,
                file_name=fil_name,
                mime='text/csv',
                key=key
            )
        
        display_data_preview("previw data", df, f"{ticker}_historical_data.csv", key=2)
        
        if modeling_option == "Prophet":
            st.write("Exécution du processus de modélisation Prophet... ")
            
            try:
                # Préparation des données pour Prophet
                df_model = df.reset_index()
                df_model = df_model[['Date', 'Close']].copy()
                df_model.columns = ['ds', 'y']
                df_model['ds'] = pd.to_datetime(df_model['ds'])
                
                if len(df_model) < 5:
                    st.error("Pas assez de données pour la prévision")
                else:
                    # Division train/test
                    total_days = len(df_model)
                    train_days = int(total_days * (1 - test_data_percentage))
                    
                    df_train = df_model.iloc[:train_days].copy()
                    df_test = df_model.iloc[train_days:].copy()
                    
                    st.write(f"Données d'entraînement : {len(df_train)} jours")
                    st.write(f"Données de test : {len(df_test)} jours")
                    
                    # Entraînement du modèle
                    model = Prophet(
                        changepoint_range=changepoints_range,
                        yearly_seasonality=True,
                        weekly_seasonality=True,
                        daily_seasonality=False
                    )
                    model.add_country_holidays(country_name=country_holidays)
                    model.fit(df_train)
                    
                    # Prédictions
                    future_dates = pd.date_range(
                        start=df_model['ds'].min(),
                        end=df_model['ds'].max(),
                        freq='D'
                    )
                    future_df = pd.DataFrame({'ds': future_dates})
                    forecast = model.predict(future_df)
                    
                    # Affichage des prédictions
                    st.subheader("Résultats de la Prévision")
                    
                    # Graphique des résultats
                    fig = go.Figure()
                    
                    # Données réelles
                    fig.add_trace(go.Scatter(
                        x=df_model['ds'],
                        y=df_model['y'],
                        mode='lines',
                        name='Prix réels',
                        line=dict(color='blue')
                    ))
                    
                    # Prédictions
                    fig.add_trace(go.Scatter(
                        x=forecast['ds'],
                        y=forecast['yhat'],
                        mode='lines',
                        name='Prédictions',
                        line=dict(color='red')
                    ))
                    
                    # Intervalle de confiance
                    fig.add_trace(go.Scatter(
                        x=forecast['ds'].tolist() + forecast['ds'].tolist()[::-1],
                        y=forecast['yhat_upper'].tolist() + forecast['yhat_lower'].tolist()[::-1],
                        fill='toself',
                        fillcolor='rgba(0,100,80,0.2)',
                        line=dict(color='rgba(255,255,255,0)'),
                        name='Intervalle de confiance'
                    ))
                    
                    fig.update_layout(
                        title=f"Prédictions pour {ticker}",
                        xaxis_title="Date",
                        yaxis_title="Prix",
                        hovermode='x unified'
                    )
                    st.plotly_chart(fig)
                    
                    # Prévisions futures
                    st.subheader("Prévisions Futures")
                    horizon = st.slider(
                        "Nombre de jours à prévoir",
                        min_value=7,
                        max_value=365,
                        value=30
                    )
                    
                    # Création des dates futures
                    last_date = df_model['ds'].max()
                    future_dates = pd.date_range(
                        start=last_date,
                        periods=horizon + 1,
                        freq='D'
                    )[1:]  # Exclure le premier jour qui est le dernier jour des données
                    
                    future_df = pd.DataFrame({'ds': future_dates})
                    future_forecast = model.predict(future_df)
                    
                    # Graphique des prévisions futures
                    fig = go.Figure()
                    
                    # Données historiques
                    fig.add_trace(go.Scatter(
                        x=df_model['ds'],
                        y=df_model['y'],
                        mode='lines',
                        name='Données historiques',
                        line=dict(color='blue')
                    ))
                    
                    # Prévisions
                    fig.add_trace(go.Scatter(
                        x=future_forecast['ds'],
                        y=future_forecast['yhat'],
                        mode='lines',
                        name='Prévisions',
                        line=dict(color='red')
                    ))
                    
                    # Intervalle de confiance
                    fig.add_trace(go.Scatter(
                        x=future_forecast['ds'].tolist() + future_forecast['ds'].tolist()[::-1],
                        y=future_forecast['yhat_upper'].tolist() + future_forecast['yhat_lower'].tolist()[::-1],
                        fill='toself',
                        fillcolor='rgba(0,100,80,0.2)',
                        line=dict(color='rgba(255,255,255,0)'),
                        name='Intervalle de confiance'
                    ))
                    
                    fig.update_layout(
                        title=f"Prévisions futures pour {ticker}",
                        xaxis_title="Date",
                        yaxis_title="Prix",
                        hovermode='x unified'
                    )
                    st.plotly_chart(fig)
                    
                    # Téléchargement des données
                    st.subheader("Téléchargement des Données")
                    
                    # Données historiques et prédictions
                    df_results = pd.merge(
                        df_model,
                        forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']],
                        on='ds',
                        how='left'
                    )
                    df_results.columns = ['Date', 'Prix réel', 'Prédiction', 'Borne inférieure', 'Borne supérieure']
                    
                    # Prévisions futures
                    df_future = future_forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
                    df_future.columns = ['Date', 'Prédiction', 'Borne inférieure', 'Borne supérieure']
                    
                    # Téléchargement des résultats
                    st.download_button(
                        label="Télécharger les résultats",
                        data=df_results.to_csv(index=False),
                        file_name=f"{ticker}_resultats.csv",
                        mime="text/csv"
                    )
                    
                    st.download_button(
                        label="Télécharger les prévisions futures",
                        data=df_future.to_csv(index=False),
                        file_name=f"{ticker}_previsions_futures.csv",
                        mime="text/csv"
                    )
                    
                    # Téléchargement du modèle
                    if st.button("Télécharger le modèle Prophet"):
                        with open('modele_prophet.json', 'w') as f:
                            f.write(model_to_json(model))
                        st.success("Modèle sauvegardé dans 'modele_prophet.json'")
                        
            except Exception as e:
                st.error(f"Erreur lors de la modélisation : {str(e)}")
                st.error("Détails de l'erreur pour le débogage :")
                st.write(e)
else:
    st.error("Aucune donnée n'a été chargée. Veuillez vérifier le symbole et les dates sélectionnées.")