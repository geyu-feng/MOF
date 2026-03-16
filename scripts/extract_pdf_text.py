from pathlib import Path
import sys

from pypdf import PdfReader


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    pdfs = list(root.glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError("No PDF file found in the project root.")

    pdf_path = pdfs[0]
    reader = PdfReader(str(pdf_path))
    text = "\n\n".join((page.extract_text() or "") for page in reader.pages)

    out_dir = root / "outputs"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "paper_text.txt"
    out_path.write_text(text, encoding="utf-8")

    preview = "\n".join(text.splitlines()[:40])
    sys.stdout.buffer.write(preview.encode("utf-8", errors="ignore"))
    sys.stdout.buffer.write(f"\n\nSaved to: {out_path}\n".encode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
