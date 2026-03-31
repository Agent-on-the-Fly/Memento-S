"""
Mock OTA Server for testing auto-update functionality.

Run this server to simulate OTA responses:
    python tests/auto_update/mock_ota_server.py

Then configure config.yaml:
    ota:
        url: "http://localhost:8888/api/check"
"""

from __future__ import annotations

import json
import random
import sys
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import parse_qs, urlparse

# Current version for testing
CURRENT_VERSION = "1.0.0"
AVAILABLE_VERSION = "1.1.0"

# Mock update packages for different platforms
UPDATE_PACKAGES = {
    "darwin": {
        "url": "http://localhost:8888/download/memento-s-v1.1.0-macos.zip",
        "filename": "memento-s-v1.1.0-macos.zip",
        "size": 25_165_824,  # ~24 MB
        "checksum": "a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6",  # Mock MD5
    },
    "windows": {
        "url": "http://localhost:8888/download/memento-s-v1.1.0-windows.zip",
        "filename": "memento-s-v1.1.0-windows.zip",
        "size": 23_068_672,  # ~22 MB
        "checksum": "b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6q7",
    },
    "linux": {
        "url": "http://localhost:8888/download/memento-s-v1.1.0-linux.tar.gz",
        "filename": "memento-s-v1.1.0-linux.tar.gz",
        "size": 22_020_096,  # ~21 MB
        "checksum": "c3d4e5f6g7h8i9j0k1l2m3n4o5p6q7r8",
    },
}


class MockOTAHandler(BaseHTTPRequestHandler):
    """Handler for mock OTA server requests."""

    def log_message(self, format, *args):
        """Custom logging."""
        print(f"[MockOTA] {format % args}")

    def do_GET(self):
        """Handle GET requests."""
        parsed = urlparse(self.path)
        path = parsed.path
        query = parse_qs(parsed.query)

        if path == "/api/check":
            self._handle_check_update(query)
        elif path.startswith("/download/"):
            self._handle_download(path)
        elif path == "/":
            self._handle_index()
        else:
            self._send_error(404, "Not Found")

    def _handle_check_update(self, query: dict):
        """Handle update check request."""
        current_version = query.get("current_version", ["0.0.0"])[0]
        platform = query.get("platform", ["linux"])[0].lower()

        print(f"[MockOTA] Check update request:")
        print(f"  Current version: {current_version}")
        print(f"  Platform: {platform}")

        # Always return update available for testing
        package = UPDATE_PACKAGES.get(platform, UPDATE_PACKAGES["linux"])

        response = {
            "update_available": True,
            "latest_version": AVAILABLE_VERSION,
            "download_url": package["url"],
            "release_notes": f"版本 {AVAILABLE_VERSION} 更新内容：\n\n"
            "- 新增自动更新功能\n"
            "- 优化性能\n"
            "- 修复已知问题\n",
            "published_at": "2024-01-15T10:00:00Z",
            "size": package["size"],
            "checksum": package["checksum"],
        }

        print(f"[MockOTA] Response: {json.dumps(response, indent=2)}")
        self._send_json(response)

    def _handle_download(self, path: str):
        """Handle download request - simulate file download."""
        filename = path.split("/")[-1]
        print(f"[MockOTA] Download request: {filename}")

        # Find package info
        package_info = None
        for pkg in UPDATE_PACKAGES.values():
            if pkg["filename"] == filename:
                package_info = pkg
                break

        if not package_info:
            self._send_error(404, "File not found")
            return

        # Simulate download - generate fake data
        size = package_info["size"]

        # Create fake data (just zeros for testing)
        # In real scenario, this would be actual update file
        data = b"\x00" * min(size, 1024 * 1024)  # Max 1MB for testing

        self.send_response(200)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
        self.end_headers()
        self.wfile.write(data)

        print(f"[MockOTA] Sent {len(data)} bytes")

    def _handle_index(self):
        """Handle root request - show API info."""
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Mock OTA Server</title>
    <style>
        body {{ font-family: Arial, sans-serif; max-width: 800px; margin: 40px auto; padding: 20px; }}
        pre {{ background: #f4f4f4; padding: 10px; border-radius: 4px; overflow-x: auto; }}
        .endpoint {{ background: #e8f4f8; padding: 15px; margin: 10px 0; border-radius: 4px; }}
    </style>
</head>
<body>
    <h1>Mock OTA Server</h1>
    <p>Current version: {CURRENT_VERSION}</p>
    <p>Available version: {AVAILABLE_VERSION}</p>

    <h2>Endpoints</h2>

    <div class="endpoint">
        <h3>GET /api/check</h3>
        <p>Check for updates</p>
        <p>Parameters:</p>
        <ul>
            <li>current_version: Current app version</li>
            <li>platform: darwin, windows, or linux</li>
        </ul>
    </div>

    <div class="endpoint">
        <h3>GET /download/&lt;filename&gt;</h3>
        <p>Download update package</p>
    </div>

    <h2>Test Configuration</h2>
    <p>Add to your config.yaml:</p>
    <pre>
ota:
    url: "http://localhost:8888/api/check"
    auto_check: true
    auto_download: true
    </pre>

    <h2>Available Packages</h2>
    <ul>
        <li>macOS: {UPDATE_PACKAGES["darwin"]["filename"]} ({UPDATE_PACKAGES["darwin"]["size"]} bytes)</li>
        <li>Windows: {UPDATE_PACKAGES["windows"]["filename"]} ({UPDATE_PACKAGES["windows"]["size"]} bytes)</li>
        <li>Linux: {UPDATE_PACKAGES["linux"]["filename"]} ({UPDATE_PACKAGES["linux"]["size"]} bytes)</li>
    </ul>
</body>
</html>
"""
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(html.encode())

    def _send_json(self, data: dict):
        """Send JSON response."""
        body = json.dumps(data, ensure_ascii=False).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_error(self, code: int, message: str):
        """Send error response."""
        self.send_response(code)
        self.send_header("Content-Type", "text/plain")
        self.end_headers()
        self.wfile.write(f"Error {code}: {message}".encode())


def run_server(port: int = 8888):
    """Run the mock OTA server."""
    server = HTTPServer(("", port), MockOTAHandler)
    print(f"=" * 60)
    print(f"Mock OTA Server running on http://localhost:{port}")
    print(f"=" * 60)
    print(f"\nTest commands:")
    print(
        f"  curl 'http://localhost:{port}/api/check?current_version=1.0.0&platform=darwin'"
    )
    print(f"\nConfig for testing:")
    print(f"  ota:")
    print(f"    url: 'http://localhost:{port}/api/check'")
    print(f"\nPress Ctrl+C to stop")
    print(f"=" * 60)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[MockOTA] Server stopped")
        server.shutdown()


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8888
    run_server(port)
