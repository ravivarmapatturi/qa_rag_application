import pytest
from langchain_core.documents import Document

from chunking_strategies import CHUNKING_STRATEGY

KNOWN_STRATEGIES = ["RecursiveCharacterTextSplitter", "CharacterTextSplitter", "tiktoken", "semantic"]


@pytest.mark.parametrize("strategy", KNOWN_STRATEGIES)
def test_returns_a_splitter(strategy):
    splitter = CHUNKING_STRATEGY(strategy)
    assert hasattr(splitter, "split_documents")


def test_unknown_strategy_raises():
    with pytest.raises(ValueError):
        CHUNKING_STRATEGY("not-a-real-strategy")


def test_recursive_splitter_actually_splits_long_text():
    splitter = CHUNKING_STRATEGY("RecursiveCharacterTextSplitter")
    long_text = "word " * 2000
    chunks = splitter.split_documents([Document(page_content=long_text)])
    assert len(chunks) > 1
