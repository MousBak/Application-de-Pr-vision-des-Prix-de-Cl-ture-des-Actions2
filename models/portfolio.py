from typing import Dict, Any
from datetime import datetime
import pandas as pd

class Position:
    def __init__(self, symbol: str, quantity: float = 0, entry_price: float = 0):
        self.symbol = symbol
        self.quantity = quantity
        self.entry_price = entry_price
        self.transactions = []
        
    def add_transaction(self, quantity: float, price: float, date: datetime):
        """Ajoute une transaction à l'historique"""
        self.transactions.append({
            'date': date,
            'quantity': quantity,
            'price': price
        })
        
    def calculate_metrics(self, current_price: float) -> Dict[str, float]:
        """Calcule les métriques de la position"""
        if self.quantity == 0:
            return {
                'market_value': 0.0,
                'cost_basis': 0.0,
                'unrealized_pl': 0.0,
                'unrealized_pl_pct': 0.0
            }
            
        market_value = self.quantity * current_price
        cost_basis = self.quantity * self.entry_price
        unrealized_pl = market_value - cost_basis
        unrealized_pl_pct = (unrealized_pl / cost_basis * 100) if cost_basis > 0 else 0
        
        return {
            'market_value': market_value,
            'cost_basis': cost_basis,
            'unrealized_pl': unrealized_pl,
            'unrealized_pl_pct': unrealized_pl_pct
        }

class Portfolio:
    def __init__(self, initial_cash: float = 100000):
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.positions: Dict[str, Position] = {}
        
    def buy(self, symbol: str, quantity: float, price: float, date: datetime):
        """Achète une quantité d'actions"""
        cost = quantity * price
        if cost > self.cash:
            raise ValueError("Fonds insuffisants pour l'achat")
            
        if symbol not in self.positions:
            self.positions[symbol] = Position(symbol)
            
        position = self.positions[symbol]
        
        # Mettre à jour le prix moyen d'entrée
        total_cost = (position.quantity * position.entry_price) + cost
        total_quantity = position.quantity + quantity
        position.entry_price = total_cost / total_quantity if total_quantity > 0 else 0
        
        # Mettre à jour la quantité
        position.quantity += quantity
        
        # Ajouter la transaction
        position.add_transaction(quantity, price, date)
        
        # Mettre à jour le cash
        self.cash -= cost
        
    def sell(self, symbol: str, quantity: float, price: float, date: datetime):
        """Vend une quantité d'actions"""
        if symbol not in self.positions:
            raise ValueError(f"Pas de position pour {symbol}")
            
        position = self.positions[symbol]
        if quantity > position.quantity:
            raise ValueError("Quantité insuffisante pour la vente")
            
        # Calculer le produit de la vente
        proceeds = quantity * price
        
        # Mettre à jour la quantité
        position.quantity -= quantity
        
        # Ajouter la transaction
        position.add_transaction(-quantity, price, date)
        
        # Mettre à jour le cash
        self.cash += proceeds
        
        # Supprimer la position si la quantité est nulle
        if position.quantity == 0:
            del self.positions[symbol]
            
    def get_position(self, symbol: str) -> Position:
        """Récupère une position par son symbole"""
        return self.positions.get(symbol)
            
    def get_performance_metrics(self, current_prices: Dict[str, float]) -> Dict[str, float]:
        """Calcule les métriques de performance du portfolio"""
        total_value = self.cash
        total_cost = 0
        unrealized_pl = 0
        
        for symbol, position in self.positions.items():
            if symbol in current_prices:
                metrics = position.calculate_metrics(current_prices[symbol])
                total_value += metrics['market_value']
                total_cost += metrics['cost_basis']
                unrealized_pl += metrics['unrealized_pl']
                
        return {
            'total_value': total_value,
            'cash': self.cash,
            'invested_value': total_cost,
            'unrealized_pl': unrealized_pl,
            'unrealized_pl_pct': (unrealized_pl / total_cost * 100) if total_cost > 0 else 0,
            'cash_allocation': (self.cash / total_value * 100) if total_value > 0 else 100
        }
        
    def get_positions_summary(self) -> pd.DataFrame:
        """Retourne un résumé des positions sous forme de DataFrame"""
        if not self.positions:
            return pd.DataFrame(columns=[
                'Symbol', 'Quantity', 'Entry Price', 'Market Value',
                'Cost Basis', 'Unrealized P&L', 'Unrealized P&L %'
            ])
            
        data = []
        for symbol, position in self.positions.items():
            data.append({
                'Symbol': symbol,
                'Quantity': position.quantity,
                'Entry Price': position.entry_price,
                'Cost Basis': position.quantity * position.entry_price
            })
        return pd.DataFrame(data)
