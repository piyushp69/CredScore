"""The Streamlit Community Cloud deployment runs frontend/dashboard.py; it must keep starting the app."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ENTRYPOINT = ROOT / "frontend" / "dashboard.py"


def test_cloud_entrypoint_runs_the_app(model_dir):
    # A fresh interpreter with sys.path as `streamlit run` leaves it: the entry point's folder
    # first (where frontend/dashboard.py could shadow the dashboard package) and no project root.
    script = (
        "import sys\n"
        f"sys.path[:] = [{str(ENTRYPOINT.parent)!r}] + [p for p in sys.path if p not in ('', {str(ROOT)!r})]\n"
        "from streamlit.testing.v1 import AppTest\n"
        f"at = AppTest.from_file({str(ENTRYPOINT)!r}, default_timeout=120).run()\n"
        "assert not at.exception, [e.value for e in at.exception]\n"
        "assert not at.error, [e.value for e in at.error]\n"
        "assert any('Model loaded' in s.value for s in at.success)\n"
        "print(at.title[0].value)\n"
    )
    run = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=300,
                         env=os.environ | {"CREDSCORE_MODEL_DIR": str(model_dir)})
    assert run.returncode == 0, run.stderr[-2000:]
    assert "Applicant underwriting" in run.stdout
