"""Versioned architecture documentation; no corpus, Qdrant or Ollama initialization."""
from __future__ import annotations

import streamlit as st

from dap_assistant.presentation.architecture_assets import architecture_assets

st.set_page_config(page_title='Teljes architektúra', layout='wide')
st.title('Teljes architektúra')
st.caption('Statikus rendszerterv a docs/architecture könyvtárból. '
           'Nem a legutóbbi chatfutás nyomkövetése; azt a Chatbot → Betekintés → Folyamat mutatja.')

assets = architecture_assets()
main_tab, components_tab, source_tab = st.tabs([
    'Teljes workflow', 'Részrendszerek', 'Szerkeszthető forrás',
])

with main_tab:
    st.markdown('#### Teljes Agentic RAG workflow')
    diagram = assets['svg'] if assets['svg'].is_file() else assets['png']
    if diagram.is_file():
        st.image(str(diagram), use_container_width=True)
        st.caption('Nagyítható eredeti fájl a letöltéssel nyitható meg. '
                   'Az ábra architekturális terv, nem egy konkrét futás bizonyítéka.')
    else:
        st.error('Hiányzik a teljes workflow ábrája: docs/architecture/full_workflow.svg. '
                 'Másold be a docs/architecture fájlokat a projekt gyökerébe.')
    for kind, label, mime in (
        ('svg', 'Teljes ábra letöltése (SVG)', 'image/svg+xml'),
        ('png', 'Teljes ábra letöltése (PNG)', 'image/png'),
    ):
        path = assets[kind]
        if path.is_file():
            st.download_button(label, data=path.read_bytes(), file_name=path.name,
                               mime=mime, key=f'architecture-{kind}')

with components_tab:
    st.markdown('#### A rendszer rétegei')
    st.markdown("""**1. Offline dokumentumfeldolgozás** – hivatalos HTML/PDF-források, letöltés, chunkolás, opcionális Qdrant-index.


**2. Fő LangGraph** – élethelyzet-felismerés, részfeladat-tervezés, worker-kiküldés, bizonyítékok egyesítése, kontextus, eszközök és válaszaudit.


**3. RAG algráf** – kérdésfeldolgozás, hibrid keresés, újrarangsorolás, bizonyítékvizsgálat, kontextus-előkészítés.


**4. Kérdésfüggő eszközök** – iratlista, határidő, vevői vagyonszerzési illeték; álláskeresési járadéknál adatbekérés és hivatalos kalkulátor.


**5. Állapot és kiértékelés** – részfeladatonkénti eredmények, eszközválaszok, végső válasz; különálló benchmark- és riportfolyamat.
""")
    st.info('Az architektúrában felsorolt lehetőségek nem jelentenek automatikusan '
            'végrehajtott lépést egy adott kérdésnél.')

with source_tab:
    st.markdown('#### Architektúra szerkeszthető forrásai')
    st.caption('A DOT a részletes, a Mermaid az egyszerűsített workflow forrása.')
    for kind, label, mime in (
        ('dot', 'Graphviz DOT', 'text/vnd.graphviz'),
        ('mermaid', 'Mermaid', 'text/plain'),
    ):
        path = assets[kind]
        if path.is_file():
            st.download_button(f'{label} letöltése', data=path.read_bytes(),
                               file_name=path.name, mime=mime,
                               key=f'architecture-source-{kind}')
        else:
            st.warning(f'A(z) {path.name} dokumentum nem található.')
