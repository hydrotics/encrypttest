import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable
from urllib.parse import urlsplit


class _HealthRequestHandler(BaseHTTPRequestHandler):
    server: "_HealthHTTPServer"
    server_version = "RevealBotHealth/1.0"

    def do_GET(self) -> None:
        path = urlsplit(self.path).path
        ready = self.server.readiness_check()
        if path in {"/", "/health", "/healthz"}:
            self._respond(200, {"status": "ok", "discord_ready": ready})
            return
        if path == "/readyz":
            status = 200 if ready else 503
            self._respond(status, {"status": "ready" if ready else "not_ready"})
            return
        self._respond(404, {"status": "not_found"})

    def _respond(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        return


class _HealthHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, address: tuple[str, int], readiness_check: Callable[[], bool]) -> None:
        self.readiness_check = readiness_check
        super().__init__(address, _HealthRequestHandler)


def start_health_server(readiness_check: Callable[[], bool], host: str | None = None, port: int | None = None) -> _HealthHTTPServer:
    bind_host = host or os.getenv("HEALTH_HOST", "0.0.0.0")
    bind_port = int(os.getenv("PORT", "10000")) if port is None else port
    server = _HealthHTTPServer((bind_host, bind_port), readiness_check)
    threading.Thread(target=server.serve_forever, name="health-http", daemon=True).start()
    return server
