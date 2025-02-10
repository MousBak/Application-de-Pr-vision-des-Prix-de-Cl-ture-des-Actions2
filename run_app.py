"""
Point d'entrée principal de l'application de prévision des prix des actions.
Lance l'interface Streamlit avec la configuration appropriée.
"""

import streamlit as st
from forecasting_app import main

if __name__ == "__main__":
    # Configuration de la page Streamlit
    st.set_page_config(
        page_title="Prévision des Prix de Clôture des Actions",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Lancement de l'application
    main()
