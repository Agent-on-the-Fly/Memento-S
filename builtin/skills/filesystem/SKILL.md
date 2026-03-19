---
name: filesystem
description: Direct filesystem operations (read, write, edit, list, search files). Use for any file manipulation tasks.
---

# Filesystem Skill

## Overview
Direct filesystem operations without external dependencies. Read, write, edit, list, copy, move, and search files.

## Usage

Use the available builtin tools (`list_dir`, `read_file`, `file_create`, `edit_file_by_lines`, `bash`) to perform file operations directly.

- Paths can be absolute or relative to working_dir
- Parent directories are created automatically for write operations
- For complex file operations not covered by builtin tools, use `bash`

---

## Common Recipes

### JSON

```python
import json

# Read
with open('data.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Write (pretty-printed)
with open('output.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=2, ensure_ascii=False)
```

### CSV

```python
import csv

# Read
with open('data.csv', 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        print(row)

# Write
with open('output.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=['name', 'value'])
    writer.writeheader()
    writer.writerows([{'name': 'a', 'value': 1}])
```

### YAML

```python
# Requires: pip install pyyaml
import yaml

# Read
with open('config.yaml', 'r') as f:
    data = yaml.safe_load(f)

# Write
with open('output.yaml', 'w') as f:
    yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
```

### Directory Operations

```bash
# List files recursively
find . -type f -name "*.py"

# Directory size
du -sh /path/to/dir

# Copy directory
cp -r src/ dst/

# Move/rename
mv old_name.txt new_name.txt
```

### File Search

```bash
# Search file contents (grep)
grep -r "search_term" --include="*.py" .

# Find files by name
find . -name "*.log" -mtime -7  # Modified in last 7 days

# Count lines
wc -l *.py
```

### Text Processing

```bash
# Sort and deduplicate
sort file.txt | uniq > sorted.txt

# Extract columns
cut -d',' -f1,3 data.csv

# Replace text
sed -i '' 's/old/new/g' file.txt  # macOS
sed -i 's/old/new/g' file.txt     # Linux
```
