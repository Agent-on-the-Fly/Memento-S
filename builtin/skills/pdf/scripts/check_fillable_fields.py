#!/usr/bin/env python3
import sys


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: check_fillable_fields.py <input.pdf>", file=sys.stderr)
        sys.exit(1)

    try:
        from pypdf import PdfReader
    except ImportError:
        print("Error: pypdf is not installed. Install it with: pip install pypdf", file=sys.stderr)
        sys.exit(1)

    pdf_path = sys.argv[1]

    try:
        reader = PdfReader(pdf_path)
    except FileNotFoundError:
        print(f"Error: File not found: {pdf_path}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: Failed to read PDF '{pdf_path}': {e}", file=sys.stderr)
        sys.exit(1)

    try:
        fields = reader.get_fields()
    except Exception as e:
        print(f"Error: Failed to extract fields from '{pdf_path}': {e}", file=sys.stderr)
        sys.exit(1)

    if fields:
        print("This PDF has fillable form fields")
    else:
        print("This PDF does not have fillable form fields; you will need to visually determine where to enter data")
