import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime

@dataclass
class Position:
    symbol: str
    quantity: float
    entry_price: float
    entry_date: datetime
    
class Portfolio:
    def __init__(self, initial_cash: float = 100000.0):
        self.positions: Dict[str, Position] = {}
        self.transactions: List[Dict] = []
        self.cash: float = initial_cash
        
    def add_cash(self, amount: float):
        """Ajoute du cash au portfolio"""
        self.cash += amount
        
    def buy(self, symbol: str, quantity: float, price: float, date: datetime):
        """Achète une position"""
        cost = quantity * price
        if cost > self.cash:
            raise ValueError("Pas assez de cash disponible")
            
        if symbol in self.positions:
            # Mise à jour de la position existante
            old_pos = self.positions[symbol]
            total_quantity = old_pos.quantity + quantity
            avg_price = ((old_pos.quantity * old_pos.entry_price) + cost) / total_quantity
            self.positions[symbol] = Position(symbol, total_quantity, avg_price, date)
        else:
            # Nouvelle position
            self.positions[symbol] = Position(symbol, quantity, price, date)
            
        self.cash -= cost
        self.transactions.append({
            'date': date,
            'type': 'BUY',
            'symbol': symbol,
            'quantity': quantity,
            'price': price,
            'cost': cost
        })
        
    def sell(self, symbol: str, quantity: float, price: float, date: datetime):
        """Vend une position"""
        if symbol not in self.positions:
            raise ValueError(f"Pas de position pour {symbol}")
            
        position = self.positions[symbol]
        if quantity > position.quantity:
            raise ValueError("Quantité insuffisante")
            
        proceeds = quantity * price
        self.cash += proceeds
        
        # Mise à jour de la position
        remaining_quantity = position.quantity - quantity
        if remaining_quantity == 0:
            del self.positions[symbol]
        else:
            self.positions[symbol] = Position(
                symbol,
                remaining_quantity,
                position.entry_price,
                position.entry_date
            )
            
        self.transactions.append({
            'date': date,
            'type': 'SELL',
            'symbol': symbol,
            'quantity': quantity,
            'price': price,
            'proceeds': proceeds
        })
        
    def get_position_value(self, symbol: str, current_price: float) -> float:
        """Calcule la valeur d'une position"""
        if symbol not in self.positions:
            return 0
        return self.positions[symbol].quantity * current_price
        
    def get_total_value(self, current_prices: Dict[str, float]) -> float:
        """Calcule la valeur totale du portfolio"""
        positions_value = sum(
            self.get_position_value(symbol, current_prices[symbol])
            for symbol in self.positions
        )
        return positions_value + self.cash
        
    def get_performance_metrics(self, current_prices: Dict[str, float]) -> Dict:
        """Calcule les métriques de performance"""
        total_value = self.get_total_value(current_prices)
        invested_value = sum(
            pos.quantity * pos.entry_price
            for pos in self.positions.values()
        )
        
        # Calcul des gains/pertes
        unrealized_pl = sum(
            (current_prices[symbol] - pos.entry_price) * pos.quantity
            for symbol, pos in self.positions.items()
        )
        
        # Calcul des métriques de risque
        if self.transactions:
            returns = []
            dates = []
            current_value = total_value
            for t in self.transactions:
                if t['type'] == 'BUY':
                    current_value -= t['cost']
                else:
                    current_value += t['proceeds']
                returns.append((total_value - current_value) / current_value)
                dates.append(t['date'])
                
            returns = pd.Series(returns, index=dates)
            volatility = returns.std() * np.sqrt(252)  # Annualisée
            sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252)  # Ratio de Sharpe
        else:
            volatility = 0
            sharpe_ratio = 0
            
        return {
            'total_value': total_value,
            'invested_value': invested_value,
            'cash': self.cash,
            'unrealized_pl': unrealized_pl,
            'unrealized_pl_pct': (unrealized_pl / invested_value) if invested_value > 0 else 0,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio
        }
        
    def optimize_portfolio(self, expected_returns: Dict[str, float], 
                         volatilities: Dict[str, float],
                         correlations: pd.DataFrame) -> Dict[str, float]:
        """
        Optimise l'allocation du portfolio selon la théorie moderne du portefeuille
        """
        from scipy.optimize import minimize
        
        def portfolio_volatility(weights):
            """Calcule la volatilité du portfolio"""
            weights = pd.Series(weights, index=correlations.index)
            portfolio_var = (weights @ correlations @ weights) * 252
            return np.sqrt(portfolio_var)
            
        def portfolio_return(weights):
            """Calcule le rendement attendu du portfolio"""
            return sum(w * expected_returns[s] for s, w in zip(correlations.index, weights))
            
        def objective(weights):
            """Fonction objectif à minimiser (ratio de Sharpe négatif)"""
            ret = portfolio_return(weights)
            vol = portfolio_volatility(weights)
            return -(ret - 0.02) / vol  # 0.02 est le taux sans risque
            
        n_assets = len(correlations)
        constraints = (
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # somme des poids = 1
        )
        bounds = tuple((0, 1) for _ in range(n_assets))  # 0 <= poids <= 1
        
        result = minimize(
            objective,
            x0=np.array([1/n_assets] * n_assets),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        return dict(zip(correlations.index, result.x))
