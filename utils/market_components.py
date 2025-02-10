"""
Fonctions pour récupérer les composants des indices boursiers
"""
import pandas as pd
import requests
from bs4 import BeautifulSoup
import yfinance as yf

def get_sp500_components():
    """Récupère les composants du S&P500"""
    try:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        table = soup.find('table', {'class': 'wikitable'})
        
        tickers = []
        companies = {}
        
        for row in table.find_all('tr')[1:]:
            cols = row.find_all('td')
            if len(cols) >= 2:
                ticker = cols[0].text.strip()
                company = cols[1].text.strip()
                tickers.append(ticker)
                companies[ticker] = f"{company} ({ticker})"
                
        return tickers, companies
    except:
        # Fallback to a small subset if web scraping fails
        default_tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
        return default_tickers, {t: f"{t}" for t in default_tickers}

def get_cac40_components():
    """Récupère les composants du CAC40"""
    tickers = [
        "AI.PA", "AIR.PA", "ALO.PA", "MT.AS", "ATO.PA", "CS.PA", "BNP.PA", 
        "EN.PA", "CAP.PA", "CA.PA", "ACA.PA", "BN.PA", "DSY.PA", "ENGI.PA", 
        "EL.PA", "RMS.PA", "KER.PA", "LR.PA", "OR.PA", "MC.PA", "ML.PA", 
        "ORA.PA", "RI.PA", "PUB.PA", "RNO.PA", "SAF.PA", "SGO.PA", "SAN.PA", 
        "SU.PA", "STM.PA", "TEP.PA", "HO.PA", "FP.PA", "URW.AS", "VIE.PA", 
        "DG.PA", "VIV.PA", "WLN.PA"
    ]
    companies = {t: t for t in tickers}  # Simplifié pour l'exemple
    return tickers, companies

def get_dax_components():
    """Récupère les composants du DAX"""
    tickers = [
        "ADS.DE", "ALV.DE", "BAS.DE", "BAYN.DE", "BMW.DE", "CON.DE", 
        "1COV.DE", "DBK.DE", "DPW.DE", "DTE.DE", "EOAN.DE", "FRE.DE", 
        "FME.DE", "HEI.DE", "HEN3.DE", "IFX.DE", "LIN.DE", "MRK.DE", 
        "MTX.DE", "MUV2.DE", "RWE.DE", "SAP.DE", "SIE.DE", "VOW3.DE"
    ]
    companies = {t: t for t in tickers}
    return tickers, companies

def get_ftse_components():
    """Récupère les composants du FTSE100"""
    tickers = [
        "AAL.L", "ABDN.L", "ADM.L", "AAF.L", "AHT.L", "ANTO.L", "AZN.L", 
        "AUTO.L", "AVV.L", "AV.L", "BAB.L", "BA.L", "BARC.L", "BDEV.L", 
        "BKG.L", "BP.L", "BRBY.L", "BT-A.L", "CCH.L", "CPG.L"
    ]
    companies = {t: t for t in tickers}
    return tickers, companies

def get_nikkei225_components():
    """Récupère les composants du Nikkei225"""
    tickers = [
        "7203.T", "9432.T", "9984.T", "6758.T", "8306.T", "6861.T", 
        "6954.T", "7974.T", "6367.T", "6501.T", "7267.T", "8316.T", 
        "8411.T", "9433.T", "6702.T", "6752.T", "7751.T", "8035.T"
    ]
    companies = {t: t for t in tickers}
    return tickers, companies
