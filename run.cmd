@echo off
pip install -r requirements.txt -q
streamlit run "titanic_webapp.py" --server.port 7021
