import pandas as pd
import numpy as np
from typing import Dict, List, Optional

class TechnicalAnalysis:
    def __init__(self, data: pd.DataFrame):
        """
        Initialise l'analyseur technique
        Args:
            data (pd.DataFrame): DataFrame avec colonnes OHLCV
        """
        self.data = data.copy()
        
        # Vérifier si les colonnes sont en anglais ou en français
        if 'Close' in self.data.columns:
            # Renommer en français si nécessaire
            column_mapping = {
                'Open': 'Ouverture',
                'High': 'Haut',
                'Low': 'Bas',
                'Close': 'Fermeture',
                'Volume': 'Volume',
                'Dividends': 'Dividendes',
                'Stock Splits': 'Fractionnement'
            }
            self.data = self.data.rename(columns=column_mapping)
        
        # Nettoyer les noms des colonnes
        self.data.columns = self.data.columns.str.strip()
        
        # Vérifier que les colonnes nécessaires sont présentes
        required_columns = ['Ouverture', 'Haut', 'Bas', 'Fermeture', 'Volume']
        missing_columns = [col for col in required_columns if col not in self.data.columns]
        if missing_columns:
            available_cols = [f"{col}" for col in self.data.columns]
            raise ValueError(f"Colonnes manquantes : {missing_columns}. Colonnes disponibles : {available_cols}")
        
    def calculate_rsi(self, period: int = 14) -> pd.Series:
        """Calcule le RSI (Relative Strength Index)"""
        try:
            close_prices = self.data['Fermeture']
            delta = close_prices.diff()
            
            # Séparer les gains et les pertes
            gain = (delta.where(delta > 0, 0)).fillna(0)
            loss = (-delta.where(delta < 0, 0)).fillna(0)
            
            # Calculer la moyenne mobile des gains et des pertes
            avg_gain = gain.ewm(com=period-1, adjust=False).mean()
            avg_loss = loss.ewm(com=period-1, adjust=False).mean()
            
            # Éviter la division par zéro
            avg_loss = avg_loss.replace(0, 0.000001)
            
            # Calculer le RS et le RSI
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            
            return rsi
            
        except Exception as e:
            print(f"Erreur lors du calcul du RSI : {str(e)}")
            raise
            
    def calculate_macd(self, fast: int = 12, slow: int = 26, signal: int = 9) -> dict:
        """Calcule le MACD (Moving Average Convergence Divergence)"""
        try:
            # Vérifier que la colonne Fermeture existe
            if 'Fermeture' not in self.data.columns:
                raise KeyError(f"La colonne 'Fermeture' n'est pas disponible. Colonnes disponibles : {self.data.columns.tolist()}")
            
            # Calculer les moyennes mobiles exponentielles
            exp1 = self.data['Close'].ewm(span=fast, adjust=False).mean()
            exp2 = self.data['Close'].ewm(span=slow, adjust=False).mean()
            
            # Calculer le MACD et sa ligne de signal
            macd_line = exp1 - exp2
            signal_line = macd_line.ewm(span=signal, adjust=False).mean()
            
            # Calculer l'histogramme
            histogram = macd_line - signal_line
            
            return {
                'macd': macd_line,
                'signal': signal_line,
                'histogram': histogram
            }
            
        except Exception as e:
            print(f"Erreur lors du calcul du MACD : {str(e)}")
            raise
        
    def calculate_moving_averages(self, periods: List[int] = [20, 50, 200]) -> Dict[str, pd.Series]:
        """Calcule plusieurs moyennes mobiles"""
        mas = {}
        for period in periods:
            mas[f'MA_{period}'] = self.data['Fermeture'].rolling(window=period).mean()
        return mas
        
    def detect_patterns(self) -> Dict[str, List[str]]:
        """Détecte les motifs graphiques basiques"""
        patterns = {
            'support_resistance': [],
            'trends': [],
            'candlestick_patterns': []
        }
        
        # Détection des tendances
        ma20 = self.data['Fermeture'].rolling(window=20).mean()
        ma50 = self.data['Fermeture'].rolling(window=50).mean()
        
        if ma20.iloc[-1] > ma50.iloc[-1] and ma20.iloc[-2] <= ma50.iloc[-2]:
            patterns['trends'].append('Golden Cross - Signal Haussier')
        elif ma20.iloc[-1] < ma50.iloc[-1] and ma20.iloc[-2] >= ma50.iloc[-2]:
            patterns['trends'].append('Death Cross - Signal Baissier')
            
        return patterns
        
    def generate_signals(self) -> Dict[str, str]:
        """Génère des signaux d'achat/vente basés sur les indicateurs"""
        signals = {}
        
        # Signal basé sur RSI
        rsi = self.calculate_rsi()
        if rsi.iloc[-1] < 30:
            signals['RSI'] = 'Survendu - Signal d\'achat potentiel'
        elif rsi.iloc[-1] > 70:
            signals['RSI'] = 'Suracheté - Signal de vente potentiel'
            
        # Signal basé sur MACD
        macd_data = self.calculate_macd()
        if macd_data['macd'].iloc[-1] > macd_data['signal'].iloc[-1] and \
           macd_data['macd'].iloc[-2] <= macd_data['signal'].iloc[-2]:
            signals['MACD'] = 'Croisement haussier - Signal d\'achat'
        elif macd_data['macd'].iloc[-1] < macd_data['signal'].iloc[-1] and \
             macd_data['macd'].iloc[-2] >= macd_data['signal'].iloc[-2]:
            signals['MACD'] = 'Croisement baissier - Signal de vente'
            
        return signals
