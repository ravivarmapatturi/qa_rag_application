"""Regenerate the fixture PDFs from their plain-text sources.

The eval test set (eval/testset.json) asks specific factual questions
against these two synthetic, fully-fictional company policy documents.
The .txt files in eval/fixtures/sources/ are the source of truth -- run
this script after editing them to rebuild the PDFs the pipeline actually
parses in eval/run_eval.py.

    python eval/fixtures/generate_fixtures.py
"""
import os

from reportlab.lib.pagesizes import LETTER
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas

HERE = os.path.dirname(os.path.abspath(__file__))
SOURCES_DIR = os.path.join(HERE, "sources")
DOCS_DIR = os.path.join(HERE, "docs")


def text_to_pdf(src_path: str, dest_path: str) -> None:
    with open(src_path, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()

    c = canvas.Canvas(dest_path, pagesize=LETTER)
    width, height = LETTER
    left_margin = 0.75 * inch
    top_margin = height - 0.75 * inch
    line_height = 14
    font_size = 10.5
    c.setFont("Helvetica", font_size)

    y = top_margin
    for line in lines:
        if y < 0.75 * inch:
            c.showPage()
            c.setFont("Helvetica", font_size)
            y = top_margin
        c.drawString(left_margin, y, line)
        y -= line_height
    c.showPage()
    c.save()


def main() -> None:
    os.makedirs(DOCS_DIR, exist_ok=True)
    for name in os.listdir(SOURCES_DIR):
        if not name.endswith(".txt"):
            continue
        src = os.path.join(SOURCES_DIR, name)
        dest = os.path.join(DOCS_DIR, name.replace(".txt", ".pdf"))
        text_to_pdf(src, dest)
        print(f"wrote {dest}")


if __name__ == "__main__":
    main()
