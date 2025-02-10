"""
Module de génération de rapports PDF pour l'analyse des actions.
"""

from fpdf import FPDF
import os
from datetime import datetime
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import numpy as np
from PIL import Image
import io
import logging

# Configuration des logs
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ReportGenerator:
    def __init__(self, data: pd.DataFrame, symbol: str):
        """
        Initialise le générateur de rapports
        Args:
            data (pd.DataFrame): DataFrame avec les données de l'action
            symbol (str): Symbole de l'action
        """
        if data is None:
            raise ValueError("Les données ne peuvent pas être None")
        if not isinstance(data, pd.DataFrame):
            raise ValueError(f"Les données doivent être un DataFrame, reçu: {type(data)}")
        if len(data) == 0:
            raise ValueError("Le DataFrame est vide")
            
        self.data = data.copy()
        self.symbol = symbol
        
        # Créer le dossier des rapports s'il n'existe pas
        self.reports_dir = "rapports"
        os.makedirs(self.reports_dir, exist_ok=True)
        
        # Vérifier que nous avons toutes les colonnes nécessaires
        required_columns = ['Ouverture', 'Haut', 'Bas', 'Fermeture', 'Volume']
        missing_columns = [col for col in required_columns if col not in self.data.columns]
        if missing_columns:
            raise ValueError(f"Colonnes manquantes dans les données : {missing_columns}")
        
        # Vérifier que nous avons assez de données
        if len(self.data) < 20:  # Au moins 20 jours pour les moyennes mobiles
            raise ValueError("Pas assez de données pour générer les rapports (minimum 20 jours requis)")
            
        print(f"ReportGenerator initialisé avec succès. Shape des données: {self.data.shape}")

    def _save_temp_plot(self, fig, filename):
        """Sauvegarde un graphique temporairement et retourne son chemin"""
        try:
            print(f"Tentative de sauvegarde du graphique: {filename}")
            
            # Configurer le graphique pour l'export
            fig.update_layout(
                width=800,
                height=400,
                margin=dict(l=50, r=50, t=50, b=50)
            )
            
            # Utiliser un chemin temporaire unique
            temp_path = os.path.join(self.reports_dir, filename)
            print(f"Chemin temporaire: {temp_path}")
            
            # Sauvegarder en format PNG
            fig.write_image(temp_path, format="png", engine="kaleido")
            print(f"Graphique sauvegardé avec succès")
            
            # Vérifier que le fichier existe
            if os.path.exists(temp_path):
                return temp_path
            print(f"ERREUR: Le fichier n'existe pas après la sauvegarde: {temp_path}")
            return None
            
        except Exception as e:
            print(f"Erreur lors de la sauvegarde du graphique {filename}: {str(e)}")
            print(f"Type d'erreur: {type(e)}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            return None

    def generate_summary_report(self):
        """Génère un rapport résumé"""
        try:
            print("Début de la génération du rapport résumé")
            
            # Créer une instance de FPDF
            pdf = FPDF()
            pdf.add_page()
            
            # En-tête du rapport
            pdf.set_font("Arial", "B", 16)
            pdf.cell(0, 10, f"Résumé - {self.symbol}", ln=True, align="C")
            pdf.ln(10)
            
            # Ajouter la date du rapport
            pdf.set_font("Arial", "", 12)
            pdf.cell(0, 10, f"Date du rapport : {datetime.now().strftime('%d/%m/%Y %H:%M')}", ln=True)
            pdf.ln(10)
            
            # Informations de base
            pdf.set_font("Arial", "B", 14)
            pdf.cell(0, 10, "Informations de base", ln=True)
            pdf.ln(5)
            
            pdf.set_font("Arial", "", 12)
            current_price = self.data['Fermeture'].iloc[-1]
            price_change = ((current_price / self.data['Fermeture'].iloc[-2]) - 1) * 100
            volume = self.data['Volume'].iloc[-1]
            
            pdf.cell(0, 10, f"Prix actuel: ${current_price:.2f}", ln=True)
            pdf.cell(0, 10, f"Variation: {price_change:.2f}%", ln=True)
            pdf.cell(0, 10, f"Volume: {volume:,.0f}", ln=True)
            pdf.ln(10)
            
            print("Création du graphique des prix")
            # Graphique des prix
            fig = go.Figure()
            fig.add_trace(go.Candlestick(
                x=self.data.index,
                open=self.data['Ouverture'],
                high=self.data['Haut'],
                low=self.data['Bas'],
                close=self.data['Fermeture']
            ))
            fig.update_layout(
                title="Graphique des prix",
                xaxis_title="Date",
                yaxis_title="Prix",
                height=400
            )
            
            print("Sauvegarde du graphique")
            # Sauvegarder le graphique comme image temporaire
            temp_file = self._save_temp_plot(fig, f"temp_price_{self.symbol}.png")
            
            # Ajouter l'image au PDF si elle existe
            if temp_file and os.path.exists(temp_file):
                try:
                    print(f"Ajout de l'image au PDF: {temp_file}")
                    pdf.image(temp_file, x=10, w=190)
                    print("Image ajoutée avec succès")
                finally:
                    if os.path.exists(temp_file):
                        os.remove(temp_file)
                        print("Fichier temporaire supprimé")
            else:
                print("ERREUR: Impossible d'ajouter l'image au PDF")
            pdf.ln(10)
            
            # Statistiques
            print("Ajout des statistiques")
            pdf.set_font("Arial", "B", 14)
            pdf.cell(0, 10, "Statistiques", ln=True)
            pdf.ln(5)
            
            pdf.set_font("Arial", "", 12)
            high_52w = self.data['Haut'].rolling(window=252).max().iloc[-1]
            low_52w = self.data['Bas'].rolling(window=252).min().iloc[-1]
            avg_volume = self.data['Volume'].mean()
            
            pdf.cell(0, 10, f"Plus haut 52 semaines: ${high_52w:.2f}", ln=True)
            pdf.cell(0, 10, f"Plus bas 52 semaines: ${low_52w:.2f}", ln=True)
            pdf.cell(0, 10, f"Volume moyen: {avg_volume:,.0f}", ln=True)
            
            # Sauvegarder le rapport
            print("Sauvegarde du rapport PDF")
            output_file = self._get_output_path("resume")
            pdf.output(output_file)
            print(f"Rapport sauvegardé: {output_file}")
            
            return output_file
            
        except Exception as e:
            print(f"Erreur lors de la génération du rapport résumé : {str(e)}")
            print(f"Type d'erreur: {type(e)}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            raise

    def generate_technical_report(self):
        """Génère un rapport technique avec les indicateurs"""
        try:
            print("Début de la génération du rapport technique")
            
            # Créer un nouveau PDF
            pdf = FPDF()
            pdf.add_page()
            
            # En-tête du rapport
            pdf.set_font("Arial", "B", 16)
            pdf.cell(0, 10, f"Rapport Technique - {self.symbol}", ln=True, align="C")
            pdf.ln(10)
            
            # Ajouter la date du rapport
            pdf.set_font("Arial", "", 12)
            pdf.cell(0, 10, f"Date du rapport : {datetime.now().strftime('%d/%m/%Y %H:%M')}", ln=True)
            pdf.ln(10)
            
            print("Création du graphique des prix")
            # Graphique des prix
            fig = go.Figure()
            fig.add_trace(go.Candlestick(
                x=self.data.index,
                open=self.data['Ouverture'],
                high=self.data['Haut'],
                low=self.data['Bas'],
                close=self.data['Fermeture']
            ))
            fig.update_layout(
                title="Graphique des prix",
                xaxis_title="Date",
                yaxis_title="Prix",
                height=400
            )
            
            print("Sauvegarde du graphique")
            # Sauvegarder le graphique comme image temporaire
            temp_file = self._save_temp_plot(fig, f"temp_tech_{self.symbol}.png")
            
            # Ajouter l'image au PDF si elle existe
            if temp_file and os.path.exists(temp_file):
                try:
                    print(f"Ajout de l'image au PDF: {temp_file}")
                    pdf.image(temp_file, x=10, w=190)
                    print("Image ajoutée avec succès")
                finally:
                    if os.path.exists(temp_file):
                        os.remove(temp_file)
                        print("Fichier temporaire supprimé")
            else:
                print("ERREUR: Impossible d'ajouter l'image au PDF")
            pdf.ln(10)
            
            # Moyennes mobiles
            print("Ajout des moyennes mobiles")
            pdf.set_font("Arial", "B", 14)
            pdf.cell(0, 10, "Moyennes Mobiles", ln=True)
            pdf.ln(5)
            
            pdf.set_font("Arial", "", 12)
            ma20 = self.data['Fermeture'].rolling(window=20).mean().iloc[-1]
            ma50 = self.data['Fermeture'].rolling(window=50).mean().iloc[-1]
            ma200 = self.data['Fermeture'].rolling(window=200).mean().iloc[-1]
            
            pdf.cell(0, 10, f"MA20: {ma20:.2f}", ln=True)
            pdf.cell(0, 10, f"MA50: {ma50:.2f}", ln=True)
            pdf.cell(0, 10, f"MA200: {ma200:.2f}", ln=True)
            pdf.ln(10)
            
            # RSI
            print("Ajout du RSI")
            pdf.set_font("Arial", "B", 14)
            pdf.cell(0, 10, "RSI (Relative Strength Index)", ln=True)
            pdf.ln(5)
            
            pdf.set_font("Arial", "", 12)
            delta = self.data['Fermeture'].diff()
            gain = (delta.where(delta > 0, 0)).fillna(0)
            loss = (-delta.where(delta < 0, 0)).fillna(0)
            avg_gain = gain.rolling(window=14).mean()
            avg_loss = loss.rolling(window=14).mean()
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            current_rsi = rsi.iloc[-1]
            
            pdf.cell(0, 10, f"RSI actuel: {current_rsi:.2f}", ln=True)
            
            if current_rsi > 70:
                pdf.cell(0, 10, "Indication: Suracheté", ln=True)
            elif current_rsi < 30:
                pdf.cell(0, 10, "Indication: Survendu", ln=True)
            else:
                pdf.cell(0, 10, "Indication: Neutre", ln=True)
            pdf.ln(10)
            
            # MACD
            print("Ajout du MACD")
            pdf.set_font("Arial", "B", 14)
            pdf.cell(0, 10, "MACD", ln=True)
            pdf.ln(5)
            
            pdf.set_font("Arial", "", 12)
            exp1 = self.data['Fermeture'].ewm(span=12, adjust=False).mean()
            exp2 = self.data['Fermeture'].ewm(span=26, adjust=False).mean()
            macd = exp1 - exp2
            signal = macd.ewm(span=9, adjust=False).mean()
            histogram = macd - signal
            
            pdf.cell(0, 10, f"MACD: {macd.iloc[-1]:.2f}", ln=True)
            pdf.cell(0, 10, f"Signal: {signal.iloc[-1]:.2f}", ln=True)
            pdf.cell(0, 10, f"Histogramme: {histogram.iloc[-1]:.2f}", ln=True)
            
            # Sauvegarder le rapport
            print("Sauvegarde du rapport PDF")
            output_file = self._get_output_path("technique")
            pdf.output(output_file)
            print(f"Rapport sauvegardé: {output_file}")
            
            return output_file
            
        except Exception as e:
            print(f"Erreur lors de la génération du rapport technique : {str(e)}")
            print(f"Type d'erreur: {type(e)}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            raise

    def generate_fundamental_report(self):
        """Génère un rapport d'analyse fondamentale"""
        try:
            print("Début de la génération du rapport fondamental")
            
            # Créer une instance de FPDF
            pdf = FPDF()
            pdf.add_page()
            
            # En-tête du rapport
            pdf.set_font("Arial", "B", 16)
            pdf.cell(0, 10, f"Rapport Fondamental - {self.symbol}", ln=True, align="C")
            pdf.ln(10)
            
            # Ajouter la date du rapport
            pdf.set_font("Arial", "", 12)
            pdf.cell(0, 10, f"Date du rapport : {datetime.now().strftime('%d/%m/%Y %H:%M')}", ln=True)
            pdf.ln(10)
            
            # Statistiques de base
            print("Ajout des statistiques de base")
            pdf.set_font("Arial", "B", 14)
            pdf.cell(0, 10, "Statistiques de base", ln=True)
            pdf.ln(5)
            
            pdf.set_font("Arial", "", 12)
            current_price = self.data['Fermeture'].iloc[-1]
            market_cap = current_price * self.data['Volume'].iloc[-1]
            avg_volume = self.data['Volume'].mean()
            
            pdf.cell(0, 10, f"Prix actuel: ${current_price:.2f}", ln=True)
            pdf.cell(0, 10, f"Capitalisation boursière: ${market_cap:,.0f}", ln=True)
            pdf.cell(0, 10, f"Volume moyen: {avg_volume:,.0f}", ln=True)
            pdf.ln(10)
            
            print("Création du graphique de la capitalisation")
            # Graphique de la capitalisation
            fig = go.Figure()
            market_cap_series = self.data['Fermeture'] * self.data['Volume']
            fig.add_trace(go.Scatter(
                x=self.data.index,
                y=market_cap_series,
                name='Capitalisation'
            ))
            fig.update_layout(
                title="Évolution de la capitalisation boursière",
                xaxis_title="Date",
                yaxis_title="Capitalisation ($)",
                height=400
            )
            
            print("Sauvegarde du graphique")
            # Sauvegarder le graphique comme image temporaire
            temp_file = self._save_temp_plot(fig, f"temp_fund_{self.symbol}.png")
            
            # Ajouter l'image au PDF si elle existe
            if temp_file and os.path.exists(temp_file):
                try:
                    print(f"Ajout de l'image au PDF: {temp_file}")
                    pdf.image(temp_file, x=10, w=190)
                    print("Image ajoutée avec succès")
                finally:
                    if os.path.exists(temp_file):
                        os.remove(temp_file)
                        print("Fichier temporaire supprimé")
            else:
                print("ERREUR: Impossible d'ajouter l'image au PDF")
            pdf.ln(10)
            
            # Sauvegarder le rapport
            print("Sauvegarde du rapport PDF")
            output_file = self._get_output_path("fondamental")
            pdf.output(output_file)
            print(f"Rapport sauvegardé: {output_file}")
            
            return output_file
            
        except Exception as e:
            print(f"Erreur lors de la génération du rapport fondamental : {str(e)}")
            print(f"Type d'erreur: {type(e)}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            raise

    def _get_output_path(self, report_type):
        """Génère un chemin de fichier pour le rapport"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"rapport_{report_type}_{self.symbol}_{timestamp}.pdf"
        return os.path.join(self.reports_dir, filename)
