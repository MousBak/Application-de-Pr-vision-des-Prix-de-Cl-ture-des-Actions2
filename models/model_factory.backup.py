from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
from statsmodels.tsa.arima.model import ARIMA
from typing import Tuple

class ModelFactory:
    @staticmethod
    def create_model(model_type: str, **kwargs) -> Any:
        """
        Crée et retourne un modèle selon le type spécifié
        """
        if model_type.lower() == 'prophet':
            return ProphetModel(
                changepoint_prior_scale=kwargs.get('changepoint_prior_scale', 0.05),
                seasonality_mode=kwargs.get('seasonality_mode', 'multiplicative'),
                sequence_length=kwargs.get('sequence_length', 10)
            )
        elif model_type.lower() == 'lstm':
            return LSTMModel(**kwargs)
        elif model_type.lower() == 'xgboost':
            return XGBoostModel(**kwargs)
        elif model_type.lower() == 'arima':
            return ARIMAModel(**kwargs)
        else:
            raise ValueError(f"Type de modèle non supporté: {model_type}")

class ProphetModel:
    def __init__(self, changepoint_prior_scale: float = 0.05, seasonality_mode: str = 'multiplicative', sequence_length: int = 10, **kwargs):
        self.model = None
        self.last_date = None
        self.last_value = None
        self.sequence_length = sequence_length
        self.changepoint_prior_scale = changepoint_prior_scale
        self.seasonality_mode = seasonality_mode
        
    def prepare_data(self, dates, values) -> pd.DataFrame:
        """Prépare les données au format requis par Prophet"""
        # Supprimer le fuseau horaire des dates
        if isinstance(dates, pd.DatetimeIndex):
            if dates.tz is not None:
                dates = dates.tz_localize(None)
        
        return pd.DataFrame({
            'ds': dates,
            'y': values
        })
        
    def fit(self, data: pd.Series, **kwargs):
        """Entraîne le modèle Prophet"""
        try:
            # Sauvegarder les dernières valeurs
            self.last_date = data.index[-1]
            self.last_value = data.values[-1]
            
            # Préparer les données pour Prophet
            df = self.prepare_data(data.index, data.values)
            
            # Créer et entraîner le modèle
            self.model = Prophet(
                changepoint_prior_scale=self.changepoint_prior_scale,
                seasonality_mode=self.seasonality_mode
            )
            self.model.fit(df)
            
        except Exception as e:
            print(f"Erreur lors de l'entraînement Prophet : {str(e)}")
            # Essayer avec des paramètres par défaut
            try:
                self.model = Prophet()
                self.model.fit(df)
            except Exception as e:
                raise ValueError(f"Erreur lors de l'entraînement Prophet avec paramètres par défaut : {str(e)}")
        
    def predict(self, data: pd.Series) -> np.ndarray:
        """Fait des prédictions avec le modèle Prophet"""
        if self.model is None:
            raise ValueError("Le modèle doit d'abord être entraîné")
            
        try:
            if isinstance(data.index, pd.DatetimeIndex) and pd.isna(data).all():
                # Prévisions futures
                future_dates = self.prepare_data(data.index, np.zeros(len(data)))
                forecast = self.model.predict(future_dates)
                predictions = forecast['yhat'].values
                
                # S'assurer que nous avons le bon nombre de prédictions
                if len(predictions) > len(data):
                    predictions = predictions[:len(data)]
                elif len(predictions) < len(data):
                    # Étendre avec la dernière prédiction
                    last_pred = predictions[-1]
                    padding = np.full(len(data) - len(predictions), last_pred)
                    predictions = np.concatenate([predictions, padding])
                
                return predictions
                
            else:
                # Prédictions sur données historiques
                if len(data) < self.sequence_length:
                    raise ValueError(f"Il faut au moins {self.sequence_length} points pour faire une prédiction")
                
                # Créer un DataFrame avec les dates
                df = self.prepare_data(data.index, np.zeros(len(data)))
                
                # Faire les prédictions
                forecast = self.model.predict(df)
                predictions = forecast['yhat'].values
                
                # Créer un tableau de NaN de la même taille que les données d'entrée
                result = np.full(len(data), np.nan)
                
                # Remplir avec les prédictions à partir de sequence_length
                result[self.sequence_length:] = predictions[self.sequence_length:]
                
                return result
                
        except Exception as e:
            print(f"Erreur lors des prédictions Prophet : {str(e)}")
            if isinstance(data.index, pd.DatetimeIndex) and pd.isna(data).all():
                # Pour les prévisions futures, utiliser la dernière valeur
                return np.full(len(data), self.last_value)
            else:
                # Pour les prédictions historiques, retourner des NaN
                return np.full(len(data), np.nan)

class LSTMModel:
    def __init__(self, sequence_length: int = 10, **kwargs):
        self.sequence_length = sequence_length
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = self._build_model(**kwargs)
        self.last_sequence = None
        self.last_value = None
        
    def _build_model(self, units: int = 50, dropout: float = 0.2) -> Sequential:
        model = Sequential([
            LSTM(units=units, return_sequences=True, input_shape=(self.sequence_length, 1)),
            Dropout(dropout),
            LSTM(units=units, return_sequences=False),
            Dropout(dropout),
            Dense(units=1)
        ])
        model.compile(optimizer='adam', loss='mse')
        return model
        
    def prepare_sequences(self, data: np.ndarray) -> tuple:
        """Prépare les séquences X et y pour l'entraînement"""
        X, y = [], []
        for i in range(len(data) - self.sequence_length):
            sequence = data[i:(i + self.sequence_length)]
            target = data[i + self.sequence_length]
            X.append(sequence)
            y.append(target)
        return np.array(X), np.array(y)
        
    def fit(self, data: pd.Series, **kwargs):
        """Entraîne le modèle LSTM"""
        if len(data) <= self.sequence_length:
            raise ValueError(f"Il faut au moins {self.sequence_length + 1} points pour l'entraînement")
            
        # Normalisation des données
        values = data.values.reshape(-1, 1)
        scaled_data = self.scaler.fit_transform(values)
        
        # Préparation des séquences
        X, y = self.prepare_sequences(scaled_data)
        
        if len(X) == 0 or len(y) == 0:
            raise ValueError("Pas assez de données pour créer des séquences")
            
        # Reshape X pour LSTM [samples, time steps, features]
        X = X.reshape((len(X), self.sequence_length, 1))
        
        # Sauvegarder la dernière séquence et valeur pour les prévisions futures
        self.last_sequence = scaled_data[-self.sequence_length:]
        self.last_value = scaled_data[-1][0]
        
        # Entraînement
        default_params = {
            'epochs': 100,
            'batch_size': 32,
            'verbose': 0,
            'shuffle': False
        }
        default_params.update(kwargs)
        
        self.model.fit(X, y, **default_params)
        
    def predict(self, data: pd.Series) -> np.ndarray:
        """Fait des prédictions avec le modèle LSTM"""
        if isinstance(data.index, pd.DatetimeIndex) and pd.isna(data).all():
            # Prévisions futures
            if self.last_sequence is None or self.last_value is None:
                raise ValueError("Le modèle doit d'abord être entraîné")
                
            predictions = []
            current_sequence = self.last_sequence.copy()
            
            # Faire les prédictions une par une
            for _ in range(len(data)):
                # Reshape pour LSTM [samples, time steps, features]
                X = current_sequence.reshape((1, self.sequence_length, 1))
                
                # Prédire la prochaine valeur
                yhat = self.model.predict(X, verbose=0)
                predictions.append(yhat[0, 0])
                
                # Mettre à jour la séquence pour la prochaine prédiction
                current_sequence = np.roll(current_sequence, -1)
                current_sequence[-1] = yhat[0, 0]
            
            # Convertir les prédictions en array
            predictions = np.array(predictions).reshape(-1, 1)
            
            # Dénormaliser les prédictions
            return self.scaler.inverse_transform(predictions).flatten()
            
        else:
            if len(data) < self.sequence_length:
                raise ValueError(f"Il faut au moins {self.sequence_length} points pour faire une prédiction")
            
            # Normaliser les données
            values = data.values.reshape(-1, 1)
            scaled_data = self.scaler.transform(values)
            
            # Préparer les séquences
            X = []
            for i in range(len(scaled_data) - self.sequence_length):
                X.append(scaled_data[i:(i + self.sequence_length)])
            X = np.array(X)
            
            if len(X) == 0:
                return np.full(len(data), np.nan)
            
            # Reshape pour LSTM [samples, time steps, features]
            X = X.reshape((len(X), self.sequence_length, 1))
            
            # Faire les prédictions
            predictions = self.model.predict(X, verbose=0)
            
            # Créer un tableau de NaN de la même taille que les données d'entrée
            result = np.full(len(data), np.nan)
            
            # Remplir avec les prédictions à partir de sequence_length
            result[self.sequence_length:] = predictions.flatten()
            
            # Dénormaliser les prédictions
            result_reshaped = result.reshape(-1, 1)
            return self.scaler.inverse_transform(result_reshaped).flatten()

class XGBoostModel:
    def __init__(self, sequence_length: int = 10, **kwargs):
        self.model = None
        self.sequence_length = sequence_length
        self.scaler = MinMaxScaler()
        self.last_sequence = None
        self.default_value = None
        
    def prepare_sequences(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Prépare les séquences pour l'entraînement"""
        # Normaliser les données
        scaled_data = self.scaler.fit_transform(data.reshape(-1, 1)).flatten()
        
        # Créer les séquences
        X, y = [], []
        for i in range(len(scaled_data) - self.sequence_length):
            X.append(scaled_data[i:(i + self.sequence_length)])
            y.append(scaled_data[i + self.sequence_length])
            
        return np.array(X), np.array(y)
        
    def fit(self, data: pd.Series, **kwargs):
        """Entraîne le modèle XGBoost"""
        try:
            if len(data) <= self.sequence_length:
                raise ValueError(f"Il faut au moins {self.sequence_length + 1} points pour l'entraînement")
            
            # Sauvegarder la dernière séquence et valeur pour les prédictions futures
            self.last_sequence = data.values[-self.sequence_length:].copy()
            self.default_value = float(data.values[-1])
            
            # Préparer les données
            X, y = self.prepare_sequences(data.values)
            
            if len(X) == 0 or len(y) == 0:
                raise ValueError("Pas assez de données pour créer des séquences")
            
            # Créer et entraîner le modèle
            self.model = xgb.XGBRegressor(
                objective='reg:squarederror',
                n_estimators=100,
                learning_rate=0.1,
                max_depth=3
            )
            
            self.model.fit(X, y)
            
        except Exception as e:
            print(f"Erreur lors de l'entraînement XGBoost : {str(e)}")
            raise
            
    def predict(self, data: pd.Series) -> np.ndarray:
        """Fait des prédictions avec le modèle XGBoost"""
        if self.model is None:
            raise ValueError("Le modèle doit d'abord être entraîné")
            
        try:
            if isinstance(data.index, pd.DatetimeIndex) and pd.isna(data).all():
                # Prévisions futures
                n_steps = len(data)
                
                if self.last_sequence is None:
                    raise ValueError("Pas de dernière séquence disponible pour les prédictions")
                
                try:
                    # Normaliser la dernière séquence
                    scaled_sequence = self.scaler.transform(self.last_sequence.reshape(-1, 1)).flatten()
                    
                    # Faire les prédictions itératives
                    predictions = []
                    current_sequence = scaled_sequence.copy()
                    
                    for _ in range(n_steps):
                        # Préparer la séquence pour la prédiction
                        X = current_sequence.reshape(1, -1)
                        
                        # Faire la prédiction
                        pred = self.model.predict(X)[0]
                        predictions.append(pred)
                        
                        # Mettre à jour la séquence
                        current_sequence = np.roll(current_sequence, -1)
                        current_sequence[-1] = pred
                    
                    # Convertir les prédictions en tableau numpy
                    predictions = np.array(predictions)
                    
                    # Dénormaliser les prédictions
                    predictions = self.scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
                    
                    return predictions
                    
                except Exception as e:
                    print(f"Erreur lors des prédictions futures XGBoost : {str(e)}")
                    return np.full(n_steps, self.default_value)
                    
            else:
                # Prédictions sur données historiques
                if len(data) <= self.sequence_length:
                    raise ValueError(f"Il faut au moins {self.sequence_length + 1} points pour faire une prédiction")
                
                try:
                    # Normaliser les données
                    scaled_data = self.scaler.transform(data.values.reshape(-1, 1)).flatten()
                    
                    # Créer un tableau de NaN de la même taille que les données d'entrée
                    result = np.full(len(data), np.nan)
                    
                    # Faire les prédictions pour chaque séquence valide
                    for i in range(self.sequence_length, len(data)):
                        sequence = scaled_data[i - self.sequence_length:i]
                        X = sequence.reshape(1, -1)
                        pred = self.model.predict(X)[0]
                        result[i] = self.scaler.inverse_transform([[pred]])[0, 0]
                    
                    return result
                    
                except Exception as e:
                    print(f"Erreur lors des prédictions historiques XGBoost : {str(e)}")
                    return np.full(len(data), np.nan)
                    
        except Exception as e:
            print(f"Erreur lors des prédictions XGBoost : {str(e)}")
            if isinstance(data.index, pd.DatetimeIndex) and pd.isna(data).all():
                return np.full(len(data), self.default_value)
            else:
                return np.full(len(data), np.nan)

class ARIMAModel:
    def __init__(self, p: int = 1, d: int = 1, q: int = 1, **kwargs):
        self.order = (p, d, q)
        self.model = None
        self.sequence_length = 10  # Valeur par défaut pour la compatibilité
        self.min_samples = 30  # Nombre minimum d'échantillons requis
        self.scaler = MinMaxScaler()  # Pour normaliser les données
        self.last_values = None
        self.default_value = None
        self.train_data = None
        
    def fit(self, data: pd.Series, **kwargs):
        """Entraîne le modèle ARIMA"""
        if len(data) < self.min_samples:
            raise ValueError(f"Il faut au moins {self.min_samples} points pour l'entraînement")
            
        try:
            # Vérifier que l'index est bien un DatetimeIndex
            if not isinstance(data.index, pd.DatetimeIndex):
                raise ValueError("Les données doivent avoir un index temporel (DatetimeIndex)")
            
            # Sauvegarder les paramètres importants
            self.train_data = data.copy()
            
            # Normaliser les données
            scaled_data = self.scaler.fit_transform(data.values.reshape(-1, 1)).flatten()
            
            # Entraîner le modèle ARIMA
            self.model = ARIMA(scaled_data, order=self.order)
            self.model = self.model.fit()
            
            return self
            
        except Exception as e:
            raise Exception(f"Erreur lors de l'entraînement ARIMA : {str(e)}")
            
    def predict(self, data: pd.Series) -> np.ndarray:
        """Fait des prédictions avec le modèle ARIMA"""
        if self.model is None:
            raise ValueError("Le modèle doit d'abord être entraîné")
            
        try:
            # Pour les prédictions futures (où data contient des NaN)
            if data.isna().all():
                # Obtenir les données d'entraînement
                train_values = self.train_data.values.reshape(-1, 1)
                train_scaled = self.scaler.transform(train_values)
                
                # Créer un nouveau modèle avec toutes les données d'entraînement
                model = ARIMA(train_scaled.flatten(), order=self.order)
                model_fit = model.fit()
                
                # Faire les prédictions
                forecast = model_fit.forecast(steps=len(data))
                predictions = np.array(forecast).reshape(-1, 1)
                
                # Dénormaliser les prédictions
                return self.scaler.inverse_transform(predictions).flatten()
            
            # Pour les prédictions sur les données de test
            else:
                predictions = []
                for i in range(len(data)):
                    # Prendre toutes les données jusqu'à l'index i
                    history = pd.concat([self.train_data, data[:i]])
                    
                    # Normaliser les données
                    history_scaled = self.scaler.transform(history.values.reshape(-1, 1)).flatten()
                    
                    # Réentraîner le modèle avec les données mises à jour
                    try:
                        model = ARIMA(history_scaled, order=self.order)
                        model_fit = model.fit()
                        
                        # Faire une prédiction pour le point suivant
                        yhat = model_fit.forecast(steps=1)[0]
                        predictions.append(yhat)
                    except Exception as e:
                        predictions.append(np.nan)
                
                # Convertir en array numpy et dénormaliser
                predictions = np.array(predictions).reshape(-1, 1)
                return self.scaler.inverse_transform(predictions).flatten()
                
        except Exception as e:
            raise Exception(f"Erreur lors des prédictions ARIMA : {str(e)}")

def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Évalue les performances du modèle avec plusieurs métriques
    Args:
        y_true: Valeurs réelles
        y_pred: Valeurs prédites
    Returns:
        Dict: Dictionnaire contenant les différentes métriques
    """
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2
    }
