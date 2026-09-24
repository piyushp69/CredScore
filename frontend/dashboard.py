"""Streamlit Community Cloud entry point.

The deployed app's main file path is frontend/dashboard.py, which predates the
project restructure. This wrapper runs the real entry point, streamlit_app.py,
so the deployed URL keeps working. Locally, `streamlit run streamlit_app.py`.
"""

import os
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# `streamlit run` puts this folder first on sys.path, where this file would
# shadow the `dashboard` package, so the project root has to come before it.
if sys.path[0] != str(ROOT):
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

runpy.run_path(str(ROOT / "streamlit_app.py"), run_name="__main__")
