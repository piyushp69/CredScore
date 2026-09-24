"""CredScore dashboard.

Run with:  streamlit run streamlit_app.py
Loads the model bundle in-process (CREDSCORE_MODEL_DIR, default models/); no API server needed.
"""

import streamlit as st

from dashboard import batch, performance, portfolio, underwriting
from dashboard.common import load_service

st.set_page_config(page_title="CredScore", page_icon="💳", layout="wide")

pages = st.navigation([
    st.Page(underwriting.render, title="Underwriting", icon=":material/person_search:", url_path="underwriting",
            default=True),
    st.Page(batch.render, title="Batch scoring", icon=":material/table_view:", url_path="batch"),
    st.Page(portfolio.render, title="Portfolio insights", icon=":material/bar_chart:", url_path="portfolio"),
    st.Page(performance.render, title="Model performance", icon=":material/speed:", url_path="performance"),
])

with st.sidebar:
    st.markdown("### 💳 CredScore\nCredit risk scoring")
    service, error = load_service()
    if service is not None:
        st.success(f"Model loaded  \n`{service.model.version}`", icon=":material/check_circle:")
    else:
        st.warning("No model loaded. Run the training pipeline.", icon=":material/warning:")

pages.run()
