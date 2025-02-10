"""
Fonctions utilitaires pour le chargement et le traitement des données boursières.
Ce module fournit des fonctions pour récupérer les composants des indices boursiers
et charger les données historiques des actions.
"""

import streamlit as st
import pandas as pd
import yfinance as yf
from bs4 import BeautifulSoup
import requests
import ssl
import urllib3
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Configuration SSL pour contourner les problèmes de certificat
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
session = requests.Session()
session.verify = False
yf.set_tz_cache_location('yfinance.cache')
ssl._create_default_https_context = ssl._create_unverified_context

@st.experimental_memo
def get_sp500_components():
    """
    Récupère la liste des composants du S&P 500 depuis Wikipedia.
    
    Returns:
        tuple: (liste des symboles, dictionnaire symbole -> nom de l'entreprise)
    """
    try:
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        response = session.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        table = soup.find('table', {'class': 'wikitable'})
        df = pd.read_html(str(table))[0]
        tickers = df['Symbol'].tolist()
        tickers_companies_dict = dict(zip(df['Symbol'], df['Security']))
        return tickers, tickers_companies_dict
    except Exception as e:
        st.error(f"Erreur lors de la récupération des composants du S&P 500: {str(e)}")
        return [], {}

@st.experimental_memo
def get_dax_components():
    """
    Récupère la liste des composants du DAX depuis Wikipedia.
    
    Returns:
        tuple: (liste des symboles, dictionnaire symbole -> nom de l'entreprise)
    """
    try:
        url = 'https://en.wikipedia.org/wiki/DAX'
        response = session.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        table = soup.find('table', {'class': 'wikitable'})
        df = pd.read_html(str(table))[4]
        tickers = df['Ticker'].tolist()
        tickers_companies_dict = dict(zip(df['Ticker'], df['Company']))
        return tickers, tickers_companies_dict
    except Exception as e:
        st.error(f"Erreur lors de la récupération des composants du DAX: {str(e)}")
        return [], {}

@st.experimental_memo
def get_nikkei225_components():
    """
    Récupère la liste des composants du Nikkei 225 depuis Wikipedia.
    
    Returns:
        tuple: (liste des symboles, dictionnaire symbole -> nom de l'entreprise)
    """
    try:
        url = "https://topforeignstocks.com/indices/the-components-of-the-nikkei-225-index/"
        response = session.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        table = soup.find('table', {'class': 'table table-striped'})
        df = pd.read_html(str(table))[0]
        df['code'] = df['code'].astype(str) + '.T'
        tickers = df['code'].tolist()
        tickers_companies_dict = dict(zip(df['code'], df['Company Name']))
        return tickers, tickers_companies_dict
    except Exception as e:
        st.error(f"Erreur lors de la récupération des composants du Nikkei 225: {str(e)}")
        return [], {}

@st.experimental_memo
def get_ftse_components():
    """
    Récupère la liste des composants du FTSE 100 depuis Wikipedia.
    
    Returns:
        tuple: (liste des symboles, dictionnaire symbole -> nom de l'entreprise)
    """
    try:
        url = 'https://en.wikipedia.org/wiki/FTSE_100_Index'
        response = session.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        table = soup.find('table', {'class': 'wikitable'})
        df = pd.read_html(str(table))[1]
        df['Ticker'] = df['Ticker'] + '.L'
        tickers = df['Ticker'].tolist()
        tickers_companies_dict = dict(zip(df['Ticker'], df['Company']))
        return tickers, tickers_companies_dict
    except Exception as e:
        st.error(f"Erreur lors de la récupération des composants du FTSE 100: {str(e)}")
        return [], {}

@st.experimental_memo
def get_cac40_components():
    """
    Récupère la liste des composants du CAC 40 depuis Wikipedia.
    
    Returns:
        tuple: (liste des symboles, dictionnaire symbole -> nom de l'entreprise)
    """
    try:
        url = 'https://en.wikipedia.org/wiki/CAC_40'
        response = session.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        table = soup.find('table', {'class': 'wikitable'})
        df = pd.read_html(str(table))[4]
        df['Ticker'] = df['Ticker'] + '.PA'
        tickers = df['Ticker'].tolist()
        tickers_companies_dict = dict(zip(df['Ticker'], df['Company']))
        return tickers, tickers_companies_dict
    except Exception as e:
        st.error(f"Erreur lors de la récupération des composants du CAC 40: {str(e)}")
        return [], {}

def format_ticker_symbol(symbol, market_index):
    """
    Formate le symbole de l'action selon l'indice boursier.
    
    Args:
        symbol (str): Symbole de l'action
        market_index (str): Nom de l'indice boursier
    
    Returns:
        str: Symbole formaté
    """
    if market_index == "S&P500":
        return symbol.replace('.','-')
    elif market_index == "CAC40":
        return f"{symbol}.PA" if not symbol.endswith('.PA') else symbol
    elif market_index == "DAX":
        return f"{symbol}.DE" if not symbol.endswith('.DE') else symbol
    elif market_index == "FTSE100":
        return f"{symbol}.L" if not symbol.endswith('.L') else symbol
    elif market_index == "Nikkei225":
        return f"{symbol}.T" if not symbol.endswith('.T') else symbol
    return symbol

@st.experimental_memo
def load_data(symbole, start, end, market_index=None):
    """
    Charge les données historiques d'une action depuis Yahoo Finance.
    
    Args:
        symbole (str): Symbole de l'action
        start (date): Date de début
        end (date): Date de fin
        market_index (str, optional): Nom de l'indice boursier
    
    Returns:
        DataFrame: Données historiques de l'action
    """
    try:
        formatted_symbol = format_ticker_symbol(symbole, market_index) if market_index else symbole
        st.write(f"Téléchargement des données pour le symbole : {formatted_symbol}")
        
        data = yf.download(
            formatted_symbol,
            start=start,
            end=end,
            progress=False,
            session=session
        )
        
        if data.empty:
            st.error(f"Aucune donnée trouvée pour {formatted_symbol}")
            return None
            
        st.success(f"Données téléchargées avec succès pour {formatted_symbol}")
        return data
    except Exception as e:
        st.error(f"Erreur lors du téléchargement des données pour {formatted_symbol}: {str(e)}")
        return None

@st.experimental_memo
def convert_df_to_csv(df):
  return df.to_csv().encode('utf-8')

def display_data_preview(title, dataframe, fil_name="close_stock_prices.csv", key=0 ):
  data_exp = st.expander(title)
  available_cols = dataframe.columns.tolist()
  column_to_show = data_exp.multiselect(
      "columns",
      available_cols,
      default=available_cols,
      key=key

  )

  data_exp.dataframe(dataframe[column_to_show])
  csv_file = convert_df_to_csv(dataframe[column_to_show])
  data_exp.download_button(
      label="Download data as CSV",
      data=csv_file,
      file_name=fil_name,
      mime='text/csv',
  )