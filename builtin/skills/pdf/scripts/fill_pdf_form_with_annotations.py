#!/usr/bin/env python3
import json
import sys

try:
    from pypdf import PdfReader, PdfWriter
    from pypdf.annotations import FreeText
except ImportError:
    print("Error: pypdf is not installed. Install it with: pip install pypdf", file=sys.stderr)
    sys.exit(1)




def transform_from_image_coords(bbox, image_width, image_height, pdf_width, pdf_height):
    x_scale = pdf_width / image_width
    y_scale = pdf_height / image_height

    left = bbox[0] * x_scale
    right = bbox[2] * x_scale

    top = pdf_height - (bbox[1] * y_scale)
    bottom = pdf_height - (bbox[3] * y_scale)

    return left, bottom, right, top


def transform_from_pdf_coords(bbox, pdf_height):
    left = bbox[0]
    right = bbox[2]

    pypdf_top = pdf_height - bbox[1]      
    pypdf_bottom = pdf_height - bbox[3]   

    return left, pypdf_bottom, right, pypdf_top


def fill_pdf_form(input_pdf_path, fields_json_path, output_pdf_path):

    try:
        with open(fields_json_path, "r") as f:
            fields_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: File not found: {fields_json_path}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in '{fields_json_path}': {e}", file=sys.stderr)
        sys.exit(1)

    if "form_fields" not in fields_data:
        print(f"Error: Missing 'form_fields' key in '{fields_json_path}'", file=sys.stderr)
        sys.exit(1)

    try:
        reader = PdfReader(input_pdf_path)
    except FileNotFoundError:
        print(f"Error: File not found: {input_pdf_path}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: Failed to read PDF '{input_pdf_path}': {e}", file=sys.stderr)
        sys.exit(1)

    writer = PdfWriter()
    
    writer.append(reader)
    
    pdf_dimensions = {}
    for i, page in enumerate(reader.pages):
        mediabox = page.mediabox
        pdf_dimensions[i + 1] = [mediabox.width, mediabox.height]
    
    annotations = []
    for field in fields_data["form_fields"]:
        page_num = field["page_number"]

        page_info = next((p for p in fields_data["pages"] if p["page_number"] == page_num), None)
        if page_info is None:
            print(f"Error: No page info found for page {page_num} in '{fields_json_path}'", file=sys.stderr)
            sys.exit(1)
        pdf_width, pdf_height = pdf_dimensions[page_num]

        if "pdf_width" in page_info:
            transformed_entry_box = transform_from_pdf_coords(
                field["entry_bounding_box"],
                float(pdf_height)
            )
        else:
            image_width = page_info["image_width"]
            image_height = page_info["image_height"]
            transformed_entry_box = transform_from_image_coords(
                field["entry_bounding_box"],
                image_width, image_height,
                float(pdf_width), float(pdf_height)
            )
        
        if "entry_text" not in field or "text" not in field["entry_text"]:
            continue
        entry_text = field["entry_text"]
        text = entry_text["text"]
        if not text:
            continue
        
        font_name = entry_text.get("font", "Arial")
        font_size = str(entry_text.get("font_size", 14)) + "pt"
        font_color = entry_text.get("font_color", "000000")

        annotation = FreeText(
            text=text,
            rect=transformed_entry_box,
            font=font_name,
            font_size=font_size,
            font_color=font_color,
            border_color=None,
            background_color=None,
        )
        annotations.append(annotation)
        writer.add_annotation(page_number=page_num - 1, annotation=annotation)
        
    try:
        with open(output_pdf_path, "wb") as output:
            writer.write(output)
    except Exception as e:
        print(f"Error: Failed to write output PDF '{output_pdf_path}': {e}", file=sys.stderr)
        sys.exit(1)
    
    print(f"Successfully filled PDF form and saved to {output_pdf_path}")
    print(f"Added {len(annotations)} text annotations")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: fill_pdf_form_with_annotations.py [input pdf] [fields.json] [output pdf]")
        sys.exit(1)
    input_pdf = sys.argv[1]
    fields_json = sys.argv[2]
    output_pdf = sys.argv[3]
    
    fill_pdf_form(input_pdf, fields_json, output_pdf)
