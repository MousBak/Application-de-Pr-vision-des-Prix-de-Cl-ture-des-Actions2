"""
Script de lancement de l'application de prévision des prix de clôture des actions.
"""

import streamlit.cli as stcli
import sys

if __name__ == '__main__':
    sys.argv = ["streamlit", "run", "forecasting_app.py"]
    sys.exit(stcli.main())
