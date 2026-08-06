"""
HTTP Range Request Server (`src/sonic/server.py`)

Simple HTTP server supporting 206 Partial Content byte ranges, enabling
instantaneous video seeking for large MP4 files in modern browsers.
"""

from http.server import HTTPServer, SimpleHTTPRequestHandler
import os
from pathlib import Path
import sys

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]


class RangeHTTPRequestHandler(SimpleHTTPRequestHandler):
    """HTTP Request Handler supporting 206 Partial Content byte ranges."""

    def send_head(self):
        path = self.translate_path(self.path)
        if os.path.isdir(path):
            return super().send_head()

        range_header = self.headers.get("Range")
        if not range_header or not range_header.startswith("bytes="):
            return super().send_head()

        try:
            file_size = os.path.getsize(path)
            range_val = range_header.split("=")[1]
            start_str, end_str = range_val.split("-")
            start = int(start_str) if start_str else 0
            end = int(end_str) if end_str else file_size - 1
            if end >= file_size:
                end = file_size - 1
            length = end - start + 1

            f = open(path, "rb")
            f.seek(start)

            self.send_response(206)
            self.send_header("Content-type", self.guess_type(path))
            self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
            self.send_header("Content-Length", str(length))
            self.send_header("Accept-Ranges", "bytes")
            self.end_headers()
            return f
        except Exception as e:
            print(f"Error handling range request for {path}: {e}", file=sys.stderr)
            return super().send_head()


def run_server(port: int = 8000):
    os.chdir(WORKSPACE_ROOT)
    server_address = ("0.0.0.0", port)
    httpd = HTTPServer(server_address, RangeHTTPRequestHandler)
    print(f"Serving Fake Shop Explorer on http://localhost:{port}/web/sonic_map/ (HTTP 206 Range Enabled)")
    httpd.serve_forever()


if __name__ == "__main__":
    port_arg = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    run_server(port_arg)
