---
name: pdf
description: Use this skill whenever the user wants to do anything with PDF files. This includes reading or extracting text/tables from PDFs, combining or merging multiple PDFs into one, splitting PDFs apart, rotating pages, adding watermarks, creating new PDFs, filling PDF forms, encrypting/decrypting PDFs, extracting images, and OCR on scanned PDFs to make them searchable. If the user mentions a .pdf file or asks to produce one, use this skill.
license: Proprietary. LICENSE.txt has complete terms
metadata:
  dependencies:
    - pypdf
    - pdfplumber
    - reportlab
---

# PDF Skill — Action Routing

## CRITICAL: Decide Your Action FIRST

Before doing anything, classify the user's request and follow the MANDATORY action:

| User wants to... | MANDATORY action |
|---|---|
| **Convert .md to .pdf / Generate PDF from markdown / 把 md 转成 pdf** | **MUST use `bash` to run `md_to_pdf.py` script** (see below) |
| **Create a new PDF from scratch** | Use `bash` to run a Python script with reportlab |
| **Read/extract text from a PDF** | Use `read_file` or `bash` with pdfplumber |
| **Merge/split/rotate/encrypt PDFs** | Use `bash` with pypdf |
| **Extract tables from a PDF** | Use `bash` with pdfplumber |
| **Fill a PDF form** | Read FORMS.md first |

---

## Markdown → PDF Conversion (MOST COMMON)

**ALWAYS use the `bash` tool to run the built-in script. NEVER use `read_file` for this task.**

```
bash: python <absolute_path_to_skill>/scripts/md_to_pdf.py <input.md> <output.pdf>
```

The script path will be listed under "Available Scripts" in the prompt. Use that absolute path directly.

Example:
```
bash: python /path/to/skills/pdf/scripts/md_to_pdf.py /workspace/report.md /workspace/report.pdf
```

Features: CJK support, headings, lists, code blocks, tables, bold/italic, blockquotes, horizontal rules.

If reportlab is not installed: `bash: pip install reportlab`

---

## Reading/Extracting from Existing PDFs

### Extract text
```python
import pdfplumber
with pdfplumber.open("document.pdf") as pdf:
    for page in pdf.pages:
        print(page.extract_text())
```

### Extract tables
```python
with pdfplumber.open("document.pdf") as pdf:
    for page in pdf.pages:
        for table in page.extract_tables():
            for row in table:
                print(row)
```

## Merge / Split / Rotate

### Merge
```python
from pypdf import PdfWriter, PdfReader
writer = PdfWriter()
for f in ["a.pdf", "b.pdf"]:
    for page in PdfReader(f).pages:
        writer.add_page(page)
with open("merged.pdf", "wb") as out:
    writer.write(out)
```

### Split
```python
reader = PdfReader("input.pdf")
for i, page in enumerate(reader.pages):
    w = PdfWriter()
    w.add_page(page)
    with open(f"page_{i+1}.pdf", "wb") as out:
        w.write(out)
```

### Rotate
```python
reader = PdfReader("input.pdf")
writer = PdfWriter()
page = reader.pages[0]
page.rotate(90)
writer.add_page(page)
with open("rotated.pdf", "wb") as out:
    writer.write(out)
```

## Password Protection
```python
from pypdf import PdfReader, PdfWriter
reader = PdfReader("input.pdf")
writer = PdfWriter()
for page in reader.pages:
    writer.add_page(page)
writer.encrypt("userpassword", "ownerpassword")
with open("encrypted.pdf", "wb") as out:
    writer.write(out)
```

## Create PDF from Scratch (reportlab)
```python
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

doc = SimpleDocTemplate("output.pdf", pagesize=letter)
styles = getSampleStyleSheet()
story = [Paragraph("Title", styles['Title']), Spacer(1, 12), Paragraph("Body text", styles['Normal'])]
doc.build(story)
```

**IMPORTANT**: Never use Unicode subscript/superscript characters in ReportLab. Use `<sub>` and `<super>` tags instead.

## Quick Reference

| Task | Tool | Method |
|------|------|--------|
| Markdown → PDF | **bash + md_to_pdf.py** | `python scripts/md_to_pdf.py in.md out.pdf` |
| Extract text | pdfplumber | `page.extract_text()` |
| Extract tables | pdfplumber | `page.extract_tables()` |
| Merge/split/rotate | pypdf | PdfReader + PdfWriter |
| Create from scratch | reportlab | SimpleDocTemplate |
| Fill forms | see FORMS.md | — |
| OCR scanned | pytesseract + pdf2image | Convert to image first |

## References
- FORMS.md — PDF form filling
- REFERENCE.md — Advanced features, JS libraries, troubleshooting
