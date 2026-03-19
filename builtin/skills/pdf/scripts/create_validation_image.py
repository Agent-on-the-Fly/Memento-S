#!/usr/bin/env python3
import json
import sys

try:
    from PIL import Image, ImageDraw
except ImportError:
    print("Error: Pillow is not installed. Install it with: pip install Pillow", file=sys.stderr)
    sys.exit(1)




def create_validation_image(page_number, fields_json_path, input_path, output_path):
    try:
        with open(fields_json_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: File not found: {fields_json_path}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in '{fields_json_path}': {e}", file=sys.stderr)
        sys.exit(1)

    if "form_fields" not in data:
        print(f"Error: Missing 'form_fields' key in '{fields_json_path}'", file=sys.stderr)
        sys.exit(1)

    try:
        img = Image.open(input_path)
    except FileNotFoundError:
        print(f"Error: Image file not found: {input_path}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: Failed to open image '{input_path}': {e}", file=sys.stderr)
        sys.exit(1)

    draw = ImageDraw.Draw(img)
    num_boxes = 0

    for field in data["form_fields"]:
        if field["page_number"] == page_number:
            entry_box = field['entry_bounding_box']
            label_box = field['label_bounding_box']
            draw.rectangle(entry_box, outline='red', width=2)
            draw.rectangle(label_box, outline='blue', width=2)
            num_boxes += 2

    try:
        img.save(output_path)
    except Exception as e:
        print(f"Error: Failed to save image '{output_path}': {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Created validation image at {output_path} with {num_boxes} bounding boxes")


if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: create_validation_image.py [page number] [fields.json file] [input image path] [output image path]")
        sys.exit(1)
    try:
        page_number = int(sys.argv[1])
    except ValueError:
        print(f"Error: Page number must be an integer, got '{sys.argv[1]}'", file=sys.stderr)
        sys.exit(1)
    fields_json_path = sys.argv[2]
    input_image_path = sys.argv[3]
    output_image_path = sys.argv[4]
    create_validation_image(page_number, fields_json_path, input_image_path, output_image_path)
