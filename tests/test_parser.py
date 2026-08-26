import pytest

from parser import PARSING_PDF

LIGHTWEIGHT_STRATEGIES = ["PyMuPDFLoader", "PyPDFLoader", "PDFMinerLoader", "pdfium"]


@pytest.mark.parametrize("strategy", LIGHTWEIGHT_STRATEGIES)
def test_parses_expected_facts(remote_work_pdf_path, strategy):
    docs = PARSING_PDF(strategy, remote_work_pdf_path)
    assert docs, f"{strategy} returned no documents"
    text = docs[0].page_content
    assert "90 days" in text
    assert "$750" in text


def test_unknown_strategy_raises(remote_work_pdf_path):
    with pytest.raises(ValueError):
        PARSING_PDF("not-a-real-strategy", remote_work_pdf_path)
