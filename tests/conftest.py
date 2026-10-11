"""Shared setup. Tests use a throwaway database, and never load models or call paid APIs."""

import os
import tempfile

# Settings are read when vantage is first imported, so pin them before any test module imports it.
# Environment variables also take precedence over a developer's .env file.
os.environ["DATABASE_URL"] = f"sqlite:///{tempfile.mkdtemp(prefix='vantage-tests-')}/traces.db"
os.environ["LLM_PROVIDER"] = "local"
os.environ["ANTHROPIC_API_KEY"] = ""

import pytest  # noqa: E402

from vantage.core import db  # noqa: E402
from vantage.eval import judge  # noqa: E402

db.init_db()


@pytest.fixture(autouse=True)
def _empty_trace_store():
    with db.get_db() as session:
        session.query(db.Trace).delete()


@pytest.fixture(autouse=True)
def _no_local_models(monkeypatch):
    """Keep the grounding check from loading the real NLI model when the local extra is installed."""
    monkeypatch.setattr(judge, "local_models_installed", lambda: False)
