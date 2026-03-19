#!/usr/bin/env python3
import json
import sys

try:
    from pypdf import PdfReader, PdfWriter
except ImportError:
    print("Error: pypdf is not installed. Install it with: pip install pypdf", file=sys.stderr)
    sys.exit(1)

try:
    from extract_form_field_info import get_field_info
except ImportError:
    print("Error: Could not import extract_form_field_info. Ensure extract_form_field_info.py is in the same directory.", file=sys.stderr)
    sys.exit(1)




def fill_pdf_fields(input_pdf_path: str, fields_json_path: str, output_pdf_path: str):
    try:
        with open(fields_json_path) as f:
            fields = json.load(f)
    except FileNotFoundError:
        print(f"Error: File not found: {fields_json_path}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in '{fields_json_path}': {e}", file=sys.stderr)
        sys.exit(1)

    fields_by_page = {}
    for field in fields:
        if "value" in field:
            field_id = field["field_id"]
            page = field["page"]
            if page not in fields_by_page:
                fields_by_page[page] = {}
            fields_by_page[page][field_id] = field["value"]

    try:
        reader = PdfReader(input_pdf_path)
    except FileNotFoundError:
        print(f"Error: File not found: {input_pdf_path}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: Failed to read PDF '{input_pdf_path}': {e}", file=sys.stderr)
        sys.exit(1)

    has_error = False
    field_info = get_field_info(reader)
    fields_by_ids = {f["field_id"]: f for f in field_info}
    for field in fields:
        existing_field = fields_by_ids.get(field["field_id"])
        if not existing_field:
            has_error = True
            print(f"ERROR: `{field['field_id']}` is not a valid field ID")
        elif field["page"] != existing_field["page"]:
            has_error = True
            print(f"ERROR: Incorrect page number for `{field['field_id']}` (got {field['page']}, expected {existing_field['page']})")
        else:
            if "value" in field:
                err = validation_error_for_field_value(existing_field, field["value"])
                if err:
                    print(err)
                    has_error = True
    if has_error:
        sys.exit(1)

    writer = PdfWriter(clone_from=reader)
    for page, field_values in fields_by_page.items():
        writer.update_page_form_field_values(writer.pages[page - 1], field_values, auto_regenerate=False)

    writer.set_need_appearances_writer(True)

    try:
        with open(output_pdf_path, "wb") as f:
            writer.write(f)
    except Exception as e:
        print(f"Error: Failed to write output PDF '{output_pdf_path}': {e}", file=sys.stderr)
        sys.exit(1)


def validation_error_for_field_value(field_info, field_value):
    field_type = field_info["type"]
    field_id = field_info["field_id"]
    if field_type == "checkbox":
        checked_val = field_info["checked_value"]
        unchecked_val = field_info["unchecked_value"]
        if field_value != checked_val and field_value != unchecked_val:
            return f'ERROR: Invalid value "{field_value}" for checkbox field "{field_id}". The checked value is "{checked_val}" and the unchecked value is "{unchecked_val}"'
    elif field_type == "radio_group":
        option_values = [opt["value"] for opt in field_info["radio_options"]]
        if field_value not in option_values:
            return f'ERROR: Invalid value "{field_value}" for radio group field "{field_id}". Valid values are: {option_values}' 
    elif field_type == "choice":
        choice_values = [opt["value"] for opt in field_info["choice_options"]]
        if field_value not in choice_values:
            return f'ERROR: Invalid value "{field_value}" for choice field "{field_id}". Valid values are: {choice_values}'
    return None


def monkeypatch_pydpf_method():
    from pypdf.generic import DictionaryObject
    from pypdf.constants import FieldDictionaryAttributes

    original_get_inherited = DictionaryObject.get_inherited

    def patched_get_inherited(self, key: str, default = None):
        result = original_get_inherited(self, key, default)
        if key == FieldDictionaryAttributes.Opt:
            if isinstance(result, list) and all(isinstance(v, list) and len(v) == 2 for v in result):
                result = [r[0] for r in result]
        return result

    DictionaryObject.get_inherited = patched_get_inherited


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: fill_fillable_fields.py [input pdf] [field_values.json] [output pdf]")
        sys.exit(1)
    monkeypatch_pydpf_method()
    input_pdf = sys.argv[1]
    fields_json = sys.argv[2]
    output_pdf = sys.argv[3]
    fill_pdf_fields(input_pdf, fields_json, output_pdf)
