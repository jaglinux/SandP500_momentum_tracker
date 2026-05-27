#!/usr/bin/env python3
"""
Local dev server for index.html.

  python serve.py
  python serve.py --port 8080

Opens http://127.0.0.1:8000 — refreshes manifest.json on each /api/manifest request.
"""

import argparse
import json
import os
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote

from generate_manifest import MANIFEST_PATH, build_manifest

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=SCRIPT_DIR, **kwargs)

    def end_headers(self):
        self.send_header("Cache-Control", "no-cache")
        super().end_headers()

    def do_GET(self):
        if self.path in ("/api/manifest", "/api/manifest.json"):
            manifest = build_manifest()
            with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2)
            body = json.dumps(manifest).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        # Block path traversal
        rel = unquote(self.path.split("?", 1)[0])
        if ".." in rel:
            self.send_error(403)
            return

        return super().do_GET()


def main():
    parser = argparse.ArgumentParser(description="Serve momentum tracker UI locally")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    build_manifest()
    url = f"http://{args.host}:{args.port}"
    print(f"Serving {SCRIPT_DIR}")
    print(f"Open {url}")
    print("Press Ctrl+C to stop")

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
        server.server_close()


if __name__ == "__main__":
    main()
