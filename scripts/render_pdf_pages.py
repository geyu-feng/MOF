from pathlib import Path

import fitz


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    pdf_path = next(root.glob("*.pdf"))
    out_dir = root / "outputs" / "paper_pages"
    out_dir.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(pdf_path)
    for page_number in [4, 5, 6]:
        page = doc[page_number]
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
        pix.save(out_dir / f"page_{page_number + 1}.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
