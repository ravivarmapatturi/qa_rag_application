"""Shared pytest fixtures.

Every real module in this repo transitively imports query_translation.py,
which constructs a ChatGroq client at import time -- that constructor only
validates that an API key string is *present*, not that it's valid, so a
dummy value here is enough to let everything import and construct without
needing real credentials or making a network call. Tests that need actual
model output use a FakeListChatModel / FakeListChatModel-backed fixture
instead of the real chatgpt.
"""
import os

os.environ.setdefault("GROQ_API_KEY", "test-dummy-key")
os.environ.setdefault("OPENAI_API_KEY", "test-dummy-key")

import glob

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "..", "eval", "fixtures", "docs")


@pytest.fixture(scope="session")
def fixture_pdf_paths():
    paths = sorted(glob.glob(os.path.join(FIXTURES_DIR, "*.pdf")))
    assert paths, "eval fixture PDFs not found -- run eval/fixtures/generate_fixtures.py"
    return paths


@pytest.fixture(scope="session")
def remote_work_pdf_path(fixture_pdf_paths):
    return next(p for p in fixture_pdf_paths if "remote_work_policy" in p)


@pytest.fixture
def fake_llm():
    """A FakeListChatModel is enough for plain .invoke() calls, but does not
    support .with_structured_output() realistically -- use only where a test
    doesn't need structured output."""
    return FakeListChatModel(responses=["fake response"] * 20)
