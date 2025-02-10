import yfinance as yf
import pandas as pd
from typing import Dict, Optional, List
import requests
from bs4 import BeautifulSoup
from datetime import datetime
import xml.etree.ElementTree as ET

class FundamentalAnalysis:
    def __init__(self, ticker: str):
        """
        Initialise l'analyseur fondamental
        Args:
            ticker (str): Symbole de l'action
        """
        self.ticker = ticker
        self.stock = yf.Ticker(ticker)
        
    def get_financial_ratios(self) -> Dict:
        """Récupère les principaux ratios financiers"""
        try:
            info = self.stock.info
            ratios = {
                'P/E Ratio': info.get('forwardPE', None),
                'PEG Ratio': info.get('pegRatio', None),
                'Price/Book': info.get('priceToBook', None),
                'Debt/Equity': info.get('debtToEquity', None),
                'ROE': info.get('returnOnEquity', None),
                'ROA': info.get('returnOnAssets', None),
                'Profit Margin': info.get('profitMargins', None),
                'Operating Margin': info.get('operatingMargins', None)
            }
            return {k: v for k, v in ratios.items() if v is not None}
        except Exception as e:
            return {'error': str(e)}
            
    def get_financial_statements(self) -> Dict[str, pd.DataFrame]:
        """Récupère les états financiers"""
        try:
            return {
                'Income Statement': self.stock.financials,
                'Balance Sheet': self.stock.balance_sheet,
                'Cash Flow': self.stock.cashflow
            }
        except Exception as e:
            return {'error': str(e)}
            
    def get_company_news(self, limit: int = 5) -> List[Dict]:
        """Récupère les dernières actualités de l'entreprise via Yahoo Finance"""
        try:
            # Construire l'URL de l'API Yahoo Finance
            url = f"https://query1.finance.yahoo.com/v1/finance/search"
            params = {
                'q': self.ticker,
                'quotesCount': 0,
                'newsCount': limit,
                'enableFuzzyQuery': False,
                'quotesQueryId': 'tss_match_phrase_query',
                'multiQuoteQueryId': 'multi_quote_single_token_query',
                'enableCb': True,
                'enableNavLinks': True,
                'enableEnhancedTrivialQuery': True
            }
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            # Faire la requête
            response = requests.get(url, headers=headers, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            # Extraire les actualités
            news_items = data.get('news', [])
            if news_items:
                formatted_news = []
                for item in news_items:
                    # Vérifier que nous avons un titre et un lien
                    if 'title' in item and 'link' in item:
                        # Nettoyer le titre
                        title = item['title'].strip()
                        if ' - ' in title:
                            title = title.split(' - ')[0].strip()
                            
                        # Formater la date
                        try:
                            published_time = datetime.fromtimestamp(item.get('providerPublishTime', 0))
                        except:
                            published_time = datetime.now()
                            
                        formatted_news.append({
                            'title': title,
                            'publisher': item.get('publisher', 'Yahoo Finance'),
                            'link': item['link'],
                            'date': published_time
                        })
                
                return formatted_news
            
            # Si pas d'actualités trouvées, essayer avec l'API alternative
            url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={self.ticker}&region=US&lang=en"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            feed = BeautifulSoup(response.content, 'xml')
            items = feed.find_all('item')
            
            formatted_news = []
            for item in items[:limit]:
                try:
                    title = item.title.text.strip()
                    if ' - ' in title:
                        title = title.split(' - ')[0].strip()
                        
                    link = item.link.text.strip()
                    pubdate = item.pubDate.text
                    
                    # Convertir la date
                    try:
                        date = datetime.strptime(pubdate, '%a, %d %b %Y %H:%M:%S %z')
                    except:
                        date = datetime.now()
                    
                    formatted_news.append({
                        'title': title,
                        'publisher': 'Yahoo Finance',
                        'link': link,
                        'date': date
                    })
                except Exception as e:
                    continue
            
            return formatted_news if formatted_news else [{
                'title': f"Voir les actualités de {self.ticker}",
                'publisher': "Yahoo Finance",
                'link': f"https://finance.yahoo.com/quote/{self.ticker}",
                'date': datetime.now()
            }]
            
        except Exception as e:
            print(f"Erreur lors de la récupération des actualités : {str(e)}")
            return [{
                'title': f"Voir les actualités de {self.ticker}",
                'publisher': "Yahoo Finance",
                'link': f"https://finance.yahoo.com/quote/{self.ticker}",
                'date': datetime.now()
            }]
            
    def analyze_market_sentiment(self) -> Dict:
        """Analyse le sentiment du marché"""
        sentiment = {
            'analyst_rating': None,
            'recommendation': None,
            'price_target': None,
            'overall_sentiment': None
        }
        
        try:
            info = self.stock.info
            
            # Recommandations des analystes
            sentiment['analyst_rating'] = info.get('recommendationMean')
            sentiment['recommendation'] = info.get('recommendationKey')
            sentiment['price_target'] = info.get('targetMeanPrice')
            
            # Calcul du sentiment global
            if sentiment['analyst_rating']:
                if sentiment['analyst_rating'] <= 2:
                    sentiment['overall_sentiment'] = 'Positif'
                elif sentiment['analyst_rating'] <= 3:
                    sentiment['overall_sentiment'] = 'Neutre'
                else:
                    sentiment['overall_sentiment'] = 'Négatif'
                    
        except Exception as e:
            sentiment['error'] = str(e)
            
        return sentiment
