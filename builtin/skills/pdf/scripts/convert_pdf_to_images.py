#!/usr/bin/env python3
import os
import sys


def convert(pdf_path, output_dir, max_dim=1000):
    try:
        from pdf2image import convert_from_path
    except ImportError:
        print("Error: pdf2image is not installed. Install it with: pip install pdf2image", file=sys.stderr)
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    try:
        images = convert_from_path(pdf_path, dpi=200)
    except Exception as e:
        print(f"Error: Failed to convert PDF '{pdf_path}' to images: {e}", file=sys.stderr)
        sys.exit(1)

    for i, image in enumerate(images):
        width, height = image.size
        if width > max_dim or height > max_dim:
            scale_factor = min(max_dim / width, max_dim / height)
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            image = image.resize((new_width, new_height))

        image_path = os.path.join(output_dir, f"page_{i+1}.png")
        try:
            image.save(image_path)
        except Exception as e:
            print(f"Error: Failed to save image '{image_path}': {e}", file=sys.stderr)
            sys.exit(1)
        print(f"Saved page {i+1} as {image_path} (size: {image.size})")

    print(f"Converted {len(images)} pages to PNG images")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: convert_pdf_to_images.py [input pdf] [output directory]")
        sys.exit(1)
    pdf_path = sys.argv[1]
    output_directory = sys.argv[2]
    convert(pdf_path, output_directory)
