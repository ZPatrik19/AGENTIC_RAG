"""Single, explicit Hungarian navigation; ignores obsolete loose files in pages/.

Streamlit normally autodiscovers every page file, including old English pages
left behind by ZIP-overwrite installs. Explicit navigation prevents duplicates.
"""
from pathlib import Path
import runpy

import streamlit as st

ROOT = Path(__file__).resolve().parent

st.set_page_config(page_title='DÁP Élethelyzet-asszisztens', page_icon='📄', layout='wide')

if hasattr(st, 'navigation'):
    navigation = st.navigation([
        st.Page(str(ROOT / 'ui.py'), title='Chatbot', icon=':material/chat:'),
        st.Page(str(ROOT / 'pages' / '1_Értékelés_és_teljesítmény.py'),
                title='Értékelés és teljesítmény', icon=':material/analytics:'),
        st.Page(str(ROOT / 'pages' / '2_Teljes_architektura.py'),
                title='Teljes architektúra', icon=':material/account_tree:'),
    ])
    navigation.run()
else:
    # Compatibility for old Streamlit; supported versions use explicit navigation.
    runpy.run_path(str(ROOT / 'ui.py'), run_name='__main__')
