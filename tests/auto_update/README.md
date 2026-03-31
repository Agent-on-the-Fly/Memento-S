# Auto-Update Test & Demo

This directory contains test utilities and demos for the auto-update functionality.

## Files

### `mock_ota_server.py`
A mock OTA server for testing update functionality.

**Usage:**
```bash
# Start the mock server
python tests/auto_update/mock_ota_server.py

# Or with custom port
python tests/auto_update/mock_ota_server.py 9999
```

**Features:**
- Simulates OTA API responses
- Provides download endpoints
- Web interface showing API documentation
- Supports macOS, Windows, Linux packages

**Configure for testing:**
```yaml
# config.yaml
ota:
  url: "http://localhost:8888/api/check"
  auto_check: true
  auto_download: true
```

### `demo.py`
A standalone demo that simulates the update process without network.

**Usage:**
```bash
# Run full interactive demo
python tests/auto_update/demo.py

# Quick state demo
python tests/auto_update/demo.py --quick

# Run specific step
python tests/auto_update/demo.py --step check
python tests/auto_update/demo.py --step download
python tests/auto_update/demo.py --step install
```

**Demo Features:**
- Simulates all update states
- Shows progress bars
- Interactive prompts
- No real network required

### `test_auto_update.py`
Integration tests for auto-update components.

**Usage:**
```bash
# Test update check
python tests/auto_update/test_auto_update.py check

# Test download (requires mock server)
python tests/auto_update/test_auto_update.py download

# Test full flow
python tests/auto_update/test_auto_update.py full

# Test cache management
python tests/auto_update/test_auto_update.py cache

# Interactive demo
python tests/auto_update/test_auto_update.py demo
```

## Quick Start

### 1. Start the Mock Server
```bash
python tests/auto_update/mock_ota_server.py
```

You should see:
```
============================================================
Mock OTA Server running on http://localhost:8888
============================================================
```

### 2. Test with Browser
Open http://localhost:8888 to see the API documentation.

Test the API:
```bash
curl 'http://localhost:8888/api/check?current_version=1.0.0&platform=darwin'
```

Expected response:
```json
{
  "update_available": true,
  "latest_version": "1.1.0",
  "download_url": "http://localhost:8888/download/memento-s-v1.1.0-macos.zip",
  ...
}
```

### 3. Run Visual Demo
```bash
# Terminal demo
python tests/auto_update/demo.py

# Or quick state demo
python tests/auto_update/demo.py --quick
```

### 4. Test with Real Components
```bash
# Requires mock server running
python tests/auto_update/test_auto_update.py full
```

## Testing Scenarios

### Scenario 1: Update Available
```bash
# 1. Start mock server
python tests/auto_update/mock_ota_server.py

# 2. Run test
python tests/auto_update/test_auto_update.py check
```

### Scenario 2: Download Progress
```bash
# Watch download progress
python tests/auto_update/test_auto_update.py download
```

### Scenario 3: Full Flow
```bash
# Complete update process
python tests/auto_update/test_auto_update.py full
```

### Scenario 4: Resume Download
1. Start a download
2. Stop the mock server during download
3. Restart mock server
4. Run download again - should resume

### Scenario 5: Cache Management
```bash
# Test cache operations
python tests/auto_update/test_auto_update.py cache
```

## API Endpoints

### GET /api/check
Check for available updates.

**Parameters:**
- `current_version`: Current app version (e.g., "1.0.0")
- `platform`: Target platform (darwin, windows, linux)
- `arch`: Architecture (optional, e.g., "x86_64")

**Response:**
```json
{
  "update_available": true,
  "latest_version": "1.1.0",
  "download_url": "...",
  "release_notes": "...",
  "published_at": "...",
  "size": 25165824,
  "checksum": "abc123..."
}
```

### GET /download/<filename>
Download update package.

## Mock Server Configuration

The mock server provides simulated update packages:

| Platform | Filename | Size | Format |
|----------|----------|------|--------|
| macOS | memento-s-v1.1.0-macos.zip | ~24 MB | ZIP |
| Windows | memento-s-v1.1.0-windows.zip | ~22 MB | ZIP |
| Linux | memento-s-v1.1.0-linux.tar.gz | ~21 MB | TAR.GZ |

## Troubleshooting

### Port Already in Use
```bash
# Use different port
python tests/auto_update/mock_ota_server.py 9999
```

### Import Errors
Make sure you're in the project root:
```bash
cd /Users/liuqiangbin/labs/memento_s
python tests/auto_update/demo.py
```

### Connection Refused
Ensure the mock server is running before testing.

## Customization

### Change Available Version
Edit `mock_ota_server.py`:
```python
AVAILABLE_VERSION = "1.2.0"  # Change this
```

### Add Custom Packages
Edit `UPDATE_PACKAGES` dict in `mock_ota_server.py`.

### Simulate Errors
Modify the server to return errors:
```python
# In _handle_check_update
if random.random() < 0.3:  # 30% failure rate
    self._send_error(500, "Server Error")
    return
```

## Integration with Main App

The test utilities can be used to verify the main app:

1. Start mock server
2. Update config.yaml with mock URL
3. Run main app
4. Watch auto-update behavior

## CI/CD Testing

For automated testing:

```bash
# Start server in background
python tests/auto_update/mock_ota_server.py &
SERVER_PID=$!

# Run tests
python tests/auto_update/test_auto_update.py full

# Cleanup
kill $SERVER_PID
```
